## Bachelor thesis Lukas Bierling: ReversibleDilatedFinBERT

### 1. Introduction
This repository contains the code for a reversible dilated BERT like model to understand financial text data.
Reversible blocks allow for huge memory savings. In combination with a dilated attention mechanism, the final model should be able to handle large sequences without causing GPU memory issues.


### Shapes of dilated attention
- Input of shape B,N,D
- Rearrange to account for different heads: B,N,H, D//H
- Rearrange to account for different segments/groups: B,S, N//S, H, D//H
- For each segment compute multihead attention, such that each scaled dot product of segment has shape: B,N//S,H,N//S
- Each segment has attention output of shape: B, N//S, H, D//H
- Reshape all segements to B, S*N//S, H, D//H = B, N, H, D//H which is the normal attention output
