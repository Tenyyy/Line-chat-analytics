import spacy

models = ["en_core_web_sm", "xx_ent_wiki_sm"]

for model in models:
    try:
        spacy.load(model)
    except OSError:
        spacy.cli.download(model)
        spacy.load(model)
