# A Model to Detect Self-disclosure Across Corpora

**TLDR: We developed a general model to detect self-disclosure in text. Our best-performing model can be found in the folder multi-task/roberta. Please follow the README there for further instructions.** 

Being able to reliably estimate self-disclosure -- a key component of friendship and intimacy -- from language is important for many psychology studies. We build single-task models on five self-disclosure corpora, but find that these models generalize poorly; the within-domain accuracy of predicted message-level self-disclosure of the best-performing model (mean Pearson's r=0.69) is much higher than the respective across data set accuracy (mean Pearson's r=0.32), due to both variations in the corpora (e.g., medical vs. general topics) and labelling instructions (target variables: self-disclosure, emotional disclosure, intimacy). However, some lexical features, such as expression of negative emotions and use of first person personal pronouns such as 'I' reliably predict self-disclosure across corpora. We develop a multi-task model that improves results, with an average Pearson's r of 0.37 for out-of-corpora prediction. 

This repository contains our self-disclosure models to determine self-disclosure across corpora in text. It is part of our research efforts documented in our paper [link will be published after anonymity period of conference submission is over; currently the paper is available upon request] and contains all models we have used as pickle files. Our best performing model was a linear multi-task model based on Roberta features.

### Multi-task Models

Our *best performing model* was a *linear multi-task model* based on *Roberta features*. It can be found in the subfolder /multitask/linear/multitask_linear_roberta_self-disclosure. It was trained on four different corpora [add citations]. You can also find our best-performing multi-task model based on LIWC features in /multitask/linear/multitask_linear_liwc_self-disclosure. Nonlinear multi-task models for both features can be found in /multitask/nonlinear directory respectively.

### Single-task Models

In addition to the multi-task models, we provide single-task models which performed best for the across-corpora prediction of self-disclosure based on a variety of features such as Ngrams, EmoLex, LIWC and RoBERTa features. The respective models can be found in /singletask.

## Usage

Please refer to the README's in the multi-task and single-task directories for specific usage guidelines.

### Example Usage for inference using the multi-task RoBERTa model:

```
python inference.py  --model_path multi-task/roberta/full_model.p --csv_path data/test.csv --pool cls --embed
```

## Data

Our models were trained using data sets from previous works researching the within-corpora prediction of self-disclosure. They are being linked in the data folder for the reproducability of our model.

## Citation

If you use one of our self-disclosure models in your work, please cite the following paper:

```
@inproceedings{reuel-etal-2022-measuring,
    title = "Measuring the Language of Self-Disclosure across Corpora",
    author = "Reuel, Ann-Katrin  and
      Peralta, Sebastian  and
      Sedoc, Jo{\~a}o  and
      Sherman, Garrick  and
      Ungar, Lyle",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.83",
    doi = "10.18653/v1/2022.findings-acl.83",
    pages = "1035--1047",
    abstract = "Being able to reliably estimate self-disclosure {--} a key component of friendship and intimacy {--} from language is important for many psychology studies. We build single-task models on five self-disclosure corpora, but find that these models generalize poorly; the within-domain accuracy of predicted message-level self-disclosure of the best-performing model (mean Pearson{'}s r=0.69) is much higher than the respective across data set accuracy (mean Pearson{'}s r=0.32), due to both variations in the corpora (e.g., medical vs. general topics) and labeling instructions (target variables: self-disclosure, emotional disclosure, intimacy). However, some lexical features, such as expression of negative emotions and use of first person personal pronouns such as {`}I{'} reliably predict self-disclosure across corpora. We develop a multi-task model that yields better results, with an average Pearson{'}s r of 0.37 for out-of-corpora prediction.",
}
```

## License

Licensed under a GNU General Public License v3 (GPLv3).
