default_electra_settings = {}
default_electra_settings["max_length"] = 128 * 16  # 2048
default_electra_settings["max_position_embeddings"] = 4096
default_electra_settings["per_device_train_batch_size"] = 1
default_electra_settings["per_device_eval_batch_size"] = 1
default_electra_settings["num_hidden_layers"] = 12
default_electra_settings["effective_batch_size"] = 4
default_electra_settings["lr"] = 5e-4