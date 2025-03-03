from torch import nn as nn

class PatchEmbed(nn.Module):
    """ 4D Image to Patch Embedding
    """

    def __init__(
        self,
        img_size=(96, 96, 96, 20),
        patch_size=(4, 4, 4, 1),
        in_chans=2,
        embed_dim=24,
        norm_layer=None,
        flatten=True,
        spatial_dims=3,
    ):
        assert len(patch_size) == 4, "you have to give four numbers, each corresponds h, w, d, t"
        #assert patch_size[3] == 1, "temporal axis merging is not implemented yet"

        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
            img_size[3] // patch_size[3],
        )
        self.embed_dim = embed_dim
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2] * self.grid_size[3]
        self.flatten = flatten

        self.fc = nn.Linear(in_features=in_chans * patch_size[0] * patch_size[1] * patch_size[2] * patch_size[3], out_features=embed_dim)

        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        B, C, D, H, W, T = x.shape
        assert D == self.img_size[0], f"Input image height ({D}) doesn't match model ({self.img_size[0]})."
        assert H == self.img_size[1], f"Input image width ({H}) doesn't match model ({self.img_size[1]})."
        assert W == self.img_size[2], f"Input image width ({W}) doesn't match model ({self.img_size[2]})."
        x = self.proj(x)
        if self.flatten:
            B, c, d, h, w, t = x.shape
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            x = self.norm(x)    # B L C
            x = x.permute(0, 2, 1).reshape(B, c, d, h, w, t)
        else: 
            x = self.norm(x)    # self.norm=nn.Identity(); B C D H W T
        return x

    def proj(self, x):
        B, C, D, H, W, T = x.shape
                        
        pD, pH, pW, pT = self.grid_size
        sD, sH, sW, sT = self.patch_size

        x = x.view(B, C, pD, sD, pH, sH, pW, sW, -1, sT)
        x = x.permute(0, 2, 4, 6, 8, 3, 5, 7, 9, 1).contiguous().view(-1, sD * sH * sW * sT * C)
        
        x = self.fc(x)
        x = x.view(B, pD, pH, pW, -1, self.embed_dim).contiguous()
        x = x.permute(0, 5, 1, 2, 3, 4)
        return x