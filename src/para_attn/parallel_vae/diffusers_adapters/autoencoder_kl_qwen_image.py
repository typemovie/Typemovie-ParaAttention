import functools

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLQwenImage
from diffusers.models.autoencoders.vae import DecoderOutput

import para_attn.primitives as DP
from para_attn.parallel_vae import init_parallel_vae_mesh


def send_tensor(tensor, dst, group):
    tensor = tensor.contiguous()
    dist.send_object_list([tensor.shape], dst=dst, group=group)
    dist.send(tensor, dst=dst, group=group)


def recv_tensor(src, group, device=None, dtype=None):
    objects = [None]
    dist.recv_object_list(objects, src=src, group=group)
    t = torch.empty(objects[0], device=device, dtype=dtype)
    dist.recv(t, src=src, group=group)
    return t


def parallelize_vae(vae: AutoencoderKLQwenImage, *, mesh=None):
    """
    Parallelize QwenImageVAE for distributed processing across multiple GPUs.

    Args:
        vae: AutoencoderKLQwenImage instance to parallelize
        mesh: Optional mesh configuration for distributed processing

    Returns:
        Parallelized VAE instance
    """
    mesh = init_parallel_vae_mesh(vae.device.type, mesh=mesh)

    group = DP.get_group(mesh)
    world_size = DP.get_world_size(group)
    rank = DP.get_rank(mesh)

    def blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                    y / blend_extent
            )
        return b

    def blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                    x / blend_extent
            )
        return b

    @functools.wraps(vae.__class__._encode)
    def new__encode(
            self,
            x: torch.Tensor,
            *args,
            **kwargs,
    ):
        # Set tiling parameters for QwenImageVAE
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192
        self.spatial_compression_ratio = 8

        batch_size, num_channels, num_frames, height, width = x.shape
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        # Compatibility check for tile size attributes
        if hasattr(self, "tile_sample_min_height"):
            tile_sample_min_height = self.tile_sample_min_height
        else:
            tile_sample_min_height = getattr(self, "tile_sample_min_size", 256)

        if hasattr(self, "tile_sample_min_width"):
            tile_sample_min_width = self.tile_sample_min_width
        else:
            tile_sample_min_width = getattr(self, "tile_sample_min_size", 256)

        # Distributed tile processing for encoding
        count = 0
        rows = []
        for j in range(0, height, self.tile_sample_stride_height):
            row = []
            for k in range(0, width, self.tile_sample_stride_width):
                if count % world_size == rank:
                    # Extract tile for current rank
                    tile = x[:, :, :, j: j + tile_sample_min_height, k: k + tile_sample_min_width]
                    self.clear_cache()

                    # Process tile through encoder with chunking
                    t = tile.shape[2]
                    iter_ = 1 + (t - 1) // 4
                    for i in range(iter_):
                        self._enc_conv_idx = [0]
                        if i == 0:
                            out = self.encoder(
                                tile[:, :, :1, :, :],
                                feat_cache=self._enc_feat_map,
                                feat_idx=self._enc_conv_idx
                            )
                        else:
                            out_ = self.encoder(
                                tile[:, :, 1 + 4 * (i - 1): 1 + 4 * i, :, :],
                                feat_cache=self._enc_feat_map,
                                feat_idx=self._enc_conv_idx,
                            )
                            out = torch.cat([out, out_], 2)

                    # Apply quantization convolution
                    enc = self.quant_conv(out)
                    self.clear_cache()
                else:
                    enc = None

                row.append(enc)
                count += 1
            rows.append(row)

        # Gather results from all ranks to rank 0
        if rank == 0:
            count = 0
            for i in range(len(rows)):
                for j in range(len(rows[i])):
                    if count % world_size != rank:
                        rows[i][j] = recv_tensor(count % world_size, group, device=x.device, dtype=x.dtype)
                    count += 1
        else:
            # Send processed tiles to rank 0
            for i in range(len(rows)):
                for j in range(len(rows[i])):
                    tile = rows[i][j]
                    if tile is not None:
                        send_tensor(tile, 0, group)

        # Blend and concatenate tiles on rank 0
        if rank == 0:
            result_rows = []
            for i, row in enumerate(rows):
                result_row = []
                for j, tile in enumerate(row):
                    # Apply blending for seamless tile transitions
                    if i > 0:
                        tile = blend_v(rows[i - 1][j], tile, blend_height)
                    if j > 0:
                        tile = blend_h(row[j - 1], tile, blend_width)
                    result_row.append(tile[:, :, :, :tile_latent_stride_height, :tile_latent_stride_width])
                result_rows.append(torch.cat(result_row, dim=-1))

            enc = torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]
        else:
            # Non-rank-0 processes receive final result
            enc = recv_tensor(0, group, device=x.device, dtype=x.dtype)

        # Chain propagation of results
        if rank < world_size - 1:
            send_tensor(enc, rank + 1, group)

        return enc

    vae._encode = new__encode.__get__(vae)

    @functools.wraps(vae.__class__._decode)
    def new__decode(
            self,
            z: torch.Tensor,
            *args,
            return_dict: bool = True,
            **kwargs,
    ):
        # Set tiling parameters for QwenImageVAE decoding
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192
        self.spatial_compression_ratio = 8

        batch_size, num_channels, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        # Distributed tile processing for decoding
        count = 0
        rows = []
        for j in range(0, height, tile_latent_stride_height):
            row = []
            for k in range(0, width, tile_latent_stride_width):
                if count % world_size == rank:
                    # Extract and process tile for current rank
                    tile = z[:, :, :, j: j + tile_latent_min_height, k: k + tile_latent_min_width]
                    tile = self.post_quant_conv(tile)
                    self.clear_cache()

                    # Process each frame individually
                    iter_ = tile.shape[2]
                    for i in range(iter_):
                        self._conv_idx = [0]
                        if i == 0:
                            out = self.decoder(
                                tile[:, :, i: i + 1, :, :],
                                feat_cache=self._feat_map,
                                feat_idx=self._conv_idx
                            )
                        else:
                            out_ = self.decoder(
                                tile[:, :, i: i + 1, :, :],
                                feat_cache=self._feat_map,
                                feat_idx=self._conv_idx
                            )
                            out = torch.cat([out, out_], 2)

                    # Apply output clamping
                    decoded = torch.clamp(out, min=-1.0, max=1.0)
                    self.clear_cache()
                else:
                    decoded = None

                row.append(decoded)
                count += 1
            rows.append(row)

        # Gather results from all ranks to rank 0
        if rank == 0:
            count = 0
            for i in range(len(rows)):
                for j in range(len(rows[i])):
                    if count % world_size != rank:
                        rows[i][j] = recv_tensor(count % world_size, group, device=z.device, dtype=z.dtype)
                    count += 1
        else:
            # Send processed tiles to rank 0
            for i in range(len(rows)):
                for j in range(len(rows[i])):
                    decoded = rows[i][j]
                    if decoded is not None:
                        send_tensor(decoded, 0, group)

        # Blend and concatenate tiles on rank 0
        if rank == 0:
            result_rows = []
            for i, row in enumerate(rows):
                result_row = []
                for j, tile in enumerate(row):
                    # Apply blending for seamless tile transitions
                    if i > 0:
                        tile = blend_v(rows[i - 1][j], tile, blend_height)
                    if j > 0:
                        tile = blend_h(row[j - 1], tile, blend_width)
                    result_row.append(tile[:, :, :, : self.tile_sample_stride_height, : self.tile_sample_stride_width])
                result_rows.append(torch.cat(result_row, dim=-1))

            dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]
        else:
            # Non-rank-0 processes receive final result
            dec = recv_tensor(0, group, device=z.device, dtype=z.dtype)

        # Chain propagation of results
        if rank < world_size - 1:
            send_tensor(dec, rank + 1, group)

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    vae._decode = new__decode.__get__(vae)

    return vae


# Usage example
def create_parallel_qwen_image_vae(model_path=None, mesh=None):
    """
    Create and parallelize a QwenImageVAE model.

    Args:
        model_path: Path to the model weights (optional)
        mesh: Mesh configuration for distributed processing

    Returns:
        Parallelized QwenImageVAE instance
    """
    # Load the QwenImageVAE model
    if model_path:
        vae = AutoencoderKLQwenImage.from_pretrained(model_path)
    else:
        # Create with default configuration
        vae = AutoencoderKLQwenImage()

    # Parallelize the VAE
    parallel_vae = parallelize_qwen_image_vae(vae, mesh=mesh)

    return parallel_vae