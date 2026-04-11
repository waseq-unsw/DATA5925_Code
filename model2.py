import torch
from torch import nn


class DayEmbeddingModel(nn.Module):
    '''
    0: <PAD>
    '''
    def __init__(self, embed_size):
        super(DayEmbeddingModel, self).__init__()

        self.day_embedding = nn.Embedding(
            num_embeddings=75+1,
            embedding_dim=embed_size,
        )

    def forward(self, day):
        embed = self.day_embedding(day)
        return embed


class TimeEmbeddingModel(nn.Module):
    '''
    0: <PAD>
    '''
    def __init__(self, embed_size):
        super(TimeEmbeddingModel, self).__init__()

        self.time_embedding = nn.Embedding(
            num_embeddings=48+1,
            embedding_dim=embed_size,
        )

    def forward(self, time):
        embed = self.time_embedding(time)
        return embed


class LocationXEmbeddingModel(nn.Module):
    '''
    0: <PAD>
    201: <MASK>
    '''
    def __init__(self, embed_size):
        super(LocationXEmbeddingModel, self).__init__()

        self.location_embedding = nn.Embedding(
            num_embeddings=202,
            embedding_dim=embed_size,
        )

    def forward(self, location):
        embed = self.location_embedding(location)
        return embed
    

class LocationYEmbeddingModel(nn.Module):
    '''
    0: <PAD>
    201: <MASK>
    '''
    def __init__(self, embed_size):
        super(LocationYEmbeddingModel, self).__init__()

        self.location_embedding = nn.Embedding(
            num_embeddings=202,
            embedding_dim=embed_size,
        )

    def forward(self, location):
        embed = self.location_embedding(location)
        return embed


class TimedeltaEmbeddingModel(nn.Module):
    '''
    0: <PAD>
    '''
    def __init__(self, embed_size):
        super(TimedeltaEmbeddingModel, self).__init__()

        self.timedelta_embedding = nn.Embedding(
            num_embeddings=48,
            embedding_dim=embed_size,
        )

    def forward(self, timedelta):
        embed = self.timedelta_embedding(timedelta)
        return embed

class WeekendEmbeddingModel(nn.Module):
    ''' 
    0: <PAD>, 
    1: Weekday, 
    2: Weekend 
    '''
    def __init__(self, embed_size):
        super(WeekendEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(3, embed_size)
    def forward(self, weekend): return self.embedding(weekend)

class MotifContinuousProj(nn.Module):
    ''' 
    Projects the 7-dimensional motif distribution [motif_0...motif_6] 
    into embed_size.
    '''
    def __init__(self, embed_size):
        super(MotifContinuousProj, self).__init__()
        self.proj = nn.Linear(7, embed_size) 
    def forward(self, motif): 
        return self.proj(motif)

class SpatialContinuousProj(nn.Module):
    ''' Projects [LDA_0...LDA_4, Density] into embed_size '''
    def __init__(self, embed_size):
        super(SpatialContinuousProj, self).__init__()
        self.proj = nn.Linear(6, embed_size) # 5 LDA + 1 Density
    def forward(self, lda, density):
        # Concatenate LDA and Density along the last dimension
        x = torch.cat([lda, density.unsqueeze(-1)], dim=-1)
        return self.proj(x)

class EmbeddingLayer(nn.Module):
    def __init__(self, embed_size):
        super(EmbeddingLayer, self).__init__()

        self.day_embedding = DayEmbeddingModel(embed_size)
        self.time_embedding = TimeEmbeddingModel(embed_size)
        self.location_x_embedding = LocationXEmbeddingModel(embed_size)
        self.location_y_embedding = LocationYEmbeddingModel(embed_size)
        self.timedelta_embedding = TimedeltaEmbeddingModel(embed_size)
        # --- NEW ---
        self.weekend_embedding = WeekendEmbeddingModel(embed_size)
        self.motif_proj = MotifContinuousProj(embed_size)
        self.spatial_proj = SpatialContinuousProj(embed_size)
        
        # --- THE NEW FUSION LAYER ---
        # We have 8 embeddings, so we concatenate them to 8 * embed_size
        # Then we project them back down to embed_size for the transformer
        self.feature_fusion = nn.Sequential(
            nn.Linear(8 * embed_size, embed_size),
            nn.LayerNorm(embed_size),  # Stabilizes the massive concatenated values
            nn.GELU()                  # Adds a slight non-linearity to help feature mixing
        )

    def forward(self, day, time, location_x, location_y, timedelta, lda, density, weekend, motif):
        day_embed = self.day_embedding(day)
        time_embed = self.time_embedding(time)
        location_x_embed = self.location_x_embedding(location_x)
        location_y_embed = self.location_y_embedding(location_y)
        timedelta_embed = self.timedelta_embedding(timedelta)
        # --- NEW ---
        weekend_embed = self.weekend_embedding(weekend)
        motif_embed = self.motif_proj(motif)
        spatial_embed = self.spatial_proj(lda, density)

        # --- THE NEW CONCATENATION ---
        # Concatenate along the last dimension (dim=-1)
        concat_embed = torch.cat([
            day_embed, time_embed, location_x_embed, location_y_embed, 
            timedelta_embed, weekend_embed, motif_embed, spatial_embed
        ], dim=-1)

        final_embed = self.feature_fusion(concat_embed)
        
        return final_embed


class TransformerEncoderModel(nn.Module):
    def __init__(self, layers_num, heads_num, embed_size):
        super(TransformerEncoderModel, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=heads_num, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=layers_num)

    def forward(self, input, src_key_padding_mask):
        out = self.transformer_encoder(input, src_key_padding_mask=src_key_padding_mask)
        return out
        

class FFNLayer(nn.Module):
    def __init__(self, embed_size):
        super(FFNLayer, self).__init__()

        self.ffn1 = nn.Sequential(
            nn.Linear(embed_size, 16),
            nn.ReLU(),
            nn.Linear(16, 200),
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(embed_size, 16),
            nn.ReLU(),
            nn.Linear(16, 200),
        )

    def forward(self, input):
        output_x = self.ffn1(input)
        output_y = self.ffn2(input)
        output = torch.stack([output_x, output_y], dim=-2)
        return output


class LPBERT(nn.Module):
    def __init__(self, layers_num, heads_num, embed_size):
        super(LPBERT, self).__init__()

        self.embedding_layer = EmbeddingLayer(embed_size)
        self.transformer_encoder = TransformerEncoderModel(layers_num, heads_num, embed_size)
        self.ffn_layer = FFNLayer(embed_size)

    def forward(self, day, time, location_x, location_y, timedelta, lda, density, weekend, motif, len):
        embed = self.embedding_layer(day, time, location_x, location_y, timedelta, lda, density, weekend, motif)
        # embed = embed.transpose(0, 1)

        max_len = day.shape[-1]
        indices = torch.arange(max_len, device=len.device).unsqueeze(0)
        src_key_padding_mask = ~(indices < len.unsqueeze(-1))

        transformer_encoder_output = self.transformer_encoder(embed, src_key_padding_mask)
        # transformer_encoder_output = transformer_encoder_output.transpose(0, 1)

        output = self.ffn_layer(transformer_encoder_output)
        return output
