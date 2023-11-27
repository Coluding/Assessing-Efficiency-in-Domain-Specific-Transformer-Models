## Bachelor thesis Lukas Bierling: Reversible dilated attention transformer to understand financial text data

### 1. Introduction


### Shapes of dilated attention
- Input of shape B,N,D
- Rearrange to account for different heads: B,N,H, D//H
- Rearrange to account for different segments/groups: B,S, N//S, H, D//H
- For each segment compute multihead attention, such that each scaled dot product of segment has shape: B,N//S,H,N//S
- Each segment has attention output of shape: B, N//S, H, D//H
- Reshape all segements to B, S*N//S, H, D//H = B, N, H, D//H which is the normal attention output