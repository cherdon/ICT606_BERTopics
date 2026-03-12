# COVID-related terms to exclude from topic terms (too frequent / non-discriminative)
COVID_STOPWORDS = [
    "covid", "covid19", "covid_19", "covid-19", "covid__19",
    "coronavirus", "pandemic", "covid19pandemic", "19"
]

# Columns to drop from the raw CSV (not needed for topic modeling)
COLUMNS_TO_DROP = [
    "id",
    "source",
    "hashtags",
    "user_mentions",
    "clean_tweet",
    "compound",
    "neg",
    "neu",
    "pos",
    "lang",
    "original_author",
]