## Bachelor thesis Lukas Bierling: Assessing Efficiency in Domain-Specific Transformer Models: Comparing, Pretraining, and Finetuning Small-Scale Transformer Models within Hardware Limitations for Financial NLP

### Abstract
This thesis embarks on a critical examination of transformer-based models optimized for the financial sector under computational constraints. The study systematically explores the adaptability and performance of various transformer architectures, including BERT, Reformer, and a custom-designed reversible dilated BERT, in handling domain-specific financial texts. A significant portion of this research is dedicated to the process of pretraining these models on specialized datasets to ascertain their effectiveness in financial sentiment analysis and topic classification tasks.

The findings indicate a nuanced landscape where the adjusted architectures, such as the Reformer and the reversible dilated BERT, show limited benefits in environments constrained by resources, particularly with smaller models and shorter sequence lengths of 128 tokens. This observation suggests that the potential advantages of these architectural modifications become more pronounced with longer sequence lengths, which were not the focus of this study due to the imposed hardware limitations.

Conversely, the Electra pretraining method, applied to the BERT model, demonstrated a promising pathway towards achieving high efficiency and robust finetuning outcomes within the specified constraints. This approach underscores the feasibility of deploying sophisticated NLP models in the financial sector, even when computational resources are limited, by leveraging domain-specific pretraining strategies.

Through a detailed analysis and comparison of transformer model architectures and their pretraining and finetuning performance, this thesis contributes valuable insights into the development of efficient NLP solutions tailored to the financial industry. It highlights the critical balance between model complexity, computational resource availability, and the specific requirements of financial NLP tasks, offering guidance for future research in this vital intersection of technology and finance.


### Repository


