from transformers import AutoModelForMaskedLM


model = AutoModelForMaskedLM.from_pretrained("yiyanghkust/finbert-pretrain")
print(model)