import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import PositionalEncoding2D, MultiHead, FeedForwardNetwork

class Encoder(nn.Module): # Inputs안의 단어는 자기만 알던 놈이었음, 이젠 남들도 아는 놈이 됨
    def __init__(self, d_model=768, h=12):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.h = h

        # nn.Embedding(단어의 개수, 임베딩할 벡터 차원)
        self.multi_head_attention = MultiHead(self.d_model, self.h) # Q단어와 K단어의 관계성에 V단어의 의미를 곱해 문장 내 모든 단어의 의미를 합친 놈이 나옴
        self.layer_norm_1 = nn.LayerNorm(self.d_model) # 정규화
        self.FFN = FeedForwardNetwork(self.d_model, self.d_model * 4) # 더 단어의 특징(의미)를 농축시킴
        self.layer_norm_2 = nn.LayerNorm(self.d_model) # 정규화

    def forward(self, Inputs, mask=None):
        norm_1 = self.layer_norm_1(Inputs)
        after_multi_head = self.multi_head_attention(norm_1, norm_1, norm_1, mask=mask)
        after_multi_head = after_multi_head + Inputs

        norm_2 = self.layer_norm_2(after_multi_head)
        after_ffn = self.FFN(norm_2)
        after_ffn = after_ffn + after_multi_head

        return after_ffn

class Decoder(nn.Module):
    def __init__(self, d_model=768, h=12):
        super(Decoder, self).__init__()
        self.h = h
        self.d_model = d_model
        
        self.self_attention = MultiHead(self.d_model, self.h)
        self.layer_norm_0 = nn.LayerNorm(self.d_model)

        self.cross_attention = MultiHead(self.d_model, self.h)
        self.layer_norm_1 = nn.LayerNorm(self.d_model)

        self.FFN = FeedForwardNetwork(self.d_model, self.d_model * 4)
        self.layer_norm_2 = nn.LayerNorm(self.d_model)

    def forward(self, p1, p2):
        norm_p1 = self.layer_norm_0(p1)
        after_self_attention = self.self_attention(norm_p1, norm_p1, norm_p1)
        after_self_attention = after_self_attention + p1

        norm_cross = self.layer_norm_1(after_self_attention)
        after_cross_attention = self.cross_attention(norm_cross, p2, p2)
        after_cross_attention = after_cross_attention + after_self_attention
        
        norm_ffn = self.layer_norm_2(after_cross_attention)
        after_ffn = self.FFN(norm_ffn)
        after_ffn = after_ffn + after_cross_attention

        return after_ffn

class CroCO(nn.Module):
    def __init__(self):
        super(CroCO, self).__init__()
        self.patch_size = 16
        self.patch_embedded_dim = 768
        self.patch_embedding = nn.Conv2d(
            in_channels= 3, 
            out_channels=self.patch_embedded_dim,
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )

        self.positionalEncoding2D = PositionalEncoding2D(d_model=768, h_patches=14, w_patches=14)

        self.encoder_layers = 12
        self.encoders = nn.ModuleList([
            Encoder(d_model=self.patch_embedded_dim, h=12) for _ in range(self.encoder_layers)
        ])

        self.decoder_layers = 8
        self.decoders = nn.ModuleList([
            Decoder(d_model=self.patch_embedded_dim, h=12) for _ in range(self.decoder_layers)
        ])

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 768))
        nn.init.normal_(self.mask_token, std=.02)

        self.predection_head = nn.Linear(768, 16 * 16 * 3)

    def forward(self, image1, image2):
        # image1 : [B, 3, H, W] [B, 3, 224, 224]
        # image2 : [B, 3, H, W] [B, 3, 224, 224]

        B = image1.shape[0]

        # [B, 3, 224, 224] -> [B, 768, 14, 14] -> [B, 768, 196] -> [B, 196, 768]
        # 196이 패치 개수고 768은 패치 하나의 임베딩 값
        p1 = self.patch_embedding(image1).flatten(2).transpose(1, 2)
        p2 = self.patch_embedding(image2).flatten(2).transpose(1, 2)

        # --- Encoder Section --- 

        p1 = self.positionalEncoding2D(p1)
        p2 = self.positionalEncoding2D(p2)

        # p1_masked [B, 19, 768]의 마스킹 안된놈들 모임
        # isMasked [B, L]로 1이면 마스킹 된놈, 0이면 마스킹 안된놈
        # ids_restore 원래 이미지의 위치
        p1_unmasked, isMasked, ids_restore = self.masking(p1, 0.9)

        for encoder in self.encoders:
            p1_unmasked = encoder(p1_unmasked)
            p2 = encoder(p2)

        # --- Decoder Section --- 
        # I'm gonna use CrossBlock to reduce computatinal load

        mask_tokens = self.mask_token.repeat(B, ids_restore.shape[1] - p1_unmasked.shape[1], 1) # [B, L-19, 768]
        x_combined = torch.cat([p1_unmasked, mask_tokens], dim=1) # [B, L, 768]
        p1_restored = torch.gather(x_combined, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_combined.shape[-1])) # [B, L, 768]
        p1_restored = self.positionalEncoding2D(p1_restored)

        x = p1_restored
        for decoder in self.decoders:
            x = decoder(x, p2)

        logits = self.predection_head(x)

        return logits, isMasked


    def masking(self, patch, ratio=0.9):
        B, L, D = patch.shape # [B, 196, 768]
        len_keep = int(L * (1 - ratio)) # 19개만 살아남을것 나머진 모두 마스킹될것

        noise = torch.rand(B, L, device=patch.device) # [B, L]의 랜덤 소수점 숫자 [0.1, 0.9, 0.2] 이런식
        ids_shuffle = torch.argsort(noise, dim=1) # [B, L]인데 순서가 인덱스로 나옴 [2, 0, 1] 이런식
        ids_restore = torch.argsort(ids_shuffle, dim=1) # [B, L]인데 순서가 인덱스로 나옴 [0, 2, 1] 이런식

        ids_keep = ids_shuffle[:, :len_keep] # [B, 0 ~ len_keep]이건 마스킹 안함, 그 뒤의 [B, len_keep ~ L]개는 마스킹 하는 놈들
        patch_masked = torch.gather(patch, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)) # ids_keep에 속하는 놈들을 모은것

        mask = torch.ones([B, L], device=patch.device) # [B, L]크기의 1
        mask[:, :len_keep] = 0 # [B, 0 ~ len_keep]은 0으로, 즉 마스킹 안된놈들은 0
        mask = torch.gather(mask, dim=1, index=ids_restore) # 원래 이미지 위치로 재배치

        # patch_masked는 [B, 19, 768]의 마스킹 안된놈들 모임
        # mask는 [B, L]로 1이면 마스킹 된놈, 0이면 마스킹 안된놈
        # ids_restore는 원래 이미지의 위치
        return patch_masked, mask, ids_restore