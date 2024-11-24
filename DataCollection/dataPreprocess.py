import pandas as pd
import re
import nltk
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
nltk.download('stopwords')

# Load dataset
data = pd.read_csv('../Datasets/spotify_songs.csv')

# Select relevant columns
columns_to_keep = [
    'track_id', 'track_name', 'track_artist', 'lyrics',
    'track_popularity', 'playlist_genre', 'playlist_subgenre', 'danceability',
    'energy', 'loudness', 'mode', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo', 'language'
]
data = data[columns_to_keep]

# Drop rows with missing values
data = data.dropna()

# Normalize numerical columns
numerical_columns = [
    'track_popularity', 'danceability', 'energy', 'loudness', 
    'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence', 'tempo'
]
scaler = MinMaxScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Feature: Lyric length (word count)
data['lyric_length'] = data['lyrics'].apply(lambda x: len(x.split()))

# Feature: Sentences in lyrics
data['sentence_count'] = data['lyrics'].apply(lambda x: len(re.split(r'[.!?]', x)))

# Analyze genre distribution (imbalance check)
genre_counts = data['playlist_genre'].value_counts()
print("Genre Distribution:\n", genre_counts)

# Advanced NLP Preprocessing: Text cleaning, stopwords removal, and lemmatization
stop_words = set(stopwords.words('english'))
def advanced_preprocess_lyrics(lyrics):
    lyrics = re.sub(r'[^\w\s]', '', lyrics)  # Remove punctuation
    lyrics = lyrics.lower()  # Convert to lowercase
    tokens = lyrics.split()  # Tokenize
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(tokens)

data['cleaned_lyrics'] = data['lyrics'].apply(advanced_preprocess_lyrics)

label_encoder_genre = LabelEncoder()
label_encoder_subgenre = LabelEncoder()

data['genre_encoded'] = label_encoder_genre.fit_transform(data['playlist_genre'])
data['subgenre_encoded'] = label_encoder_subgenre.fit_transform(data['playlist_subgenre'])

# Save mappings for later interpretation
genre_mapping = dict(zip(label_encoder_genre.classes_, label_encoder_genre.transform(label_encoder_genre.classes_)))
subgenre_mapping = dict(zip(label_encoder_subgenre.classes_, label_encoder_subgenre.transform(label_encoder_subgenre.classes_)))

print("Genre Mapping:", genre_mapping)
print("Subgenre Mapping:", subgenre_mapping)

# Save the cleaned data
data.to_csv('../Datasets/cleaned_spotify_songs.csv', index=False)

# List of words to exclude from the WordCloud
exclude_list = [
    "2g1c", "2 girls 1 cup", "acrotomophilia", "alabama hot pocket", "alaskan pipeline",
    "anal", "anilingus", "anus", "apeshit", "arsehole", "ass", "asshole", "assmunch",
    "auto erotic", "autoerotic", "babeland", "baby batter", "baby juice", "ball gag",
    "ball gravy", "ball kicking", "ball licking", "ball sack", "ball sucking", "bangbros",
    "bangbus", "bareback", "barely legal", "barenaked", "bastard", "bastardo", "bastinado",
    "bbw", "bdsm", "beaner", "beaners", "beaver cleaver", "beaver lips", "beastiality",
    "bestiality", "big black", "big breasts", "big knockers", "big tits", "bimbos", "birdlock",
    "bitch", "bitches", "black cock", "blonde action", "blonde on blonde action", "blowjob",
    "blow job", "blow your load", "blue waffle", "blumpkin", "bollocks", "bondage", "boner",
    "boob", "boobs", "booty call", "brown showers", "brunette action", "bukkake", "bulldyke",
    "bullet vibe", "bullshit", "bung hole", "bunghole", "busty", "butt", "buttcheeks", "butthole",
    "camel toe", "camgirl", "camslut", "camwhore", "carpet muncher", "carpetmuncher", "chocolate rosebuds",
    "cialis", "circlejerk", "cleveland steamer", "clit", "clitoris", "clover clamps", "clusterfuck",
    "cock", "cocks", "coprolagnia", "coprophilia", "cornhole", "coon", "coons", "creampie", "cum",
    "cumming", "cumshot", "cumshots", "cunnilingus", "cunt", "darkie", "date rape", "daterape",
    "deep throat", "deepthroat", "dendrophilia", "dick", "dildo", "dingleberry", "dingleberries",
    "dirty pillows", "dirty sanchez", "doggie style", "doggiestyle", "doggy style", "doggystyle",
    "dog style", "dolcett", "domination", "dominatrix", "dommes", "donkey punch", "double dong",
    "double penetration", "dp action", "dry hump", "dvda", "eat my ass", "ecchi", "ejaculation",
    "erotic", "erotism", "escort", "eunuch", "fag", "faggot", "fecal", "felch", "fellatio", "feltch",
    "female squirting", "femdom", "figging", "fingerbang", "fingering", "fisting", "foot fetish",
    "footjob", "frotting", "fuck", "fuck buttons", "fuckin", "fucking", "fucktards", "fudge packer",
    "fudgepacker", "futanari", "gangbang", "gang bang", "gay sex", "genitals", "giant cock",
    "girl on", "girl on top", "girls gone wild", "goatcx", "goatse", "god damn", "gokkun",
    "golden shower", "goodpoop", "goo girl", "goregasm", "grope", "group sex", "g-spot", "guro",
    "hand job", "handjob", "hard core", "hardcore", "hentai", "homoerotic", "honkey", "hooker",
    "horny", "hot carl", "hot chick", "how to kill", "how to murder", "huge fat", "humping",
    "incest", "intercourse", "jack off", "jail bait", "jailbait", "jelly donut", "jerk off", 
    "jigaboo", "jiggaboo", "jiggerboo", "jizz", "juggs", "kike", "kinbaku", "kinkster", "kinky",
    "knobbing", "leather restraint", "leather straight jacket", "lemon party", "livesex", "lolita",
    "lovemaking", "make me come", "male squirting", "masturbate", "masturbating", "masturbation",
    "menage a trois", "milf", "missionary position", "mong", "motherfucker", "mound of venus",
    "mr hands", "muff diver", "muffdiving", "nambla", "nawashi", "negro", "neonazi", "nigga",
    "nigger", "nig nog", "nimphomania", "nipple", "nipples", "nsfw", "nsfw images", "nude", 
    "nudity", "nutten", "nympho", "nymphomania", "octopussy", "omorashi", "one cup two girls",
    "one guy one jar", "orgasm", "orgy", "paedophile", "paki", "panties", "panty", "pedobear",
    "pedophile", "pegging", "penis", "phone sex", "piece of shit", "pikey", "pissing", "piss pig",
    "pisspig", "playboy", "pleasure chest", "pole smoker", "ponyplay", "poof", "poon", "poontang",
    "punany", "poop chute", "poopchute", "porn", "porno", "pornography", "prince albert piercing",
    "pthc", "pubes", "pussy", "queaf", "queef", "quim", "raghead", "raging boner", "rape", "raping",
    "rapist", "rectum", "reverse cowgirl", "rimjob", "rimming", "rosy palm", "rosy palm and her 5 sisters",
    "rusty trombone", "sadism", "santorum", "scat", "schlong", "scissoring", "semen", "sex", "sexcam",
    "sexo", "sexy", "sexual", "sexually", "sexuality", "shaved beaver", "shaved pussy", "shemale", 
    "shibari", "shit", "shitblimp", "shitty", "shota", "shrimping", "skeet", "slanteye", "slut", "s&m",
    "smut", "snatch", "snowballing", "sodomize", "sodomy", "spastic", "spic", "splooge", "splooge moose",
    "spooge", "spread legs", "spunk", "strap on", "strapon", "strappado", "strip club", "style doggy", 
    "suck", "sucks", "suicide girls", "sultry women", "swastika", "swinger", "tainted love", "taste my",
    "tea bagging", "threesome", "throating", "thumbzilla", "tied up", "tight white", "tit", "tits",
    "titties", "titty", "tongue in a", "topless", "tosser", "towelhead", "tranny", "tribadism",
    "tub girl", "tubgirl", "tushy", "twat", "twink", "twinkie", "two girls one cup", "undressing",
    "upskirt", "urethra play", "urophilia", "vagina", "venus mound", "viagra", "vibrator", "violet wand",
    "vorarephilia", "voyeur", "voyeurweb", "voyuer", "vulva", "wank", "wetback", "wet dream", 
    "white power", "whore", "worldsex", "wrapping men", "wrinkled starfish", "xx", "xxx", "yaoi", 
    "yellow showers", "yiffy", "zoophilia", "🖕"
]


def remove_excluded_words(lyrics, exclude_list):
    words = lyrics.split()
    filtered = [word for word in words if word not in exclude_list]
    return ' '.join(filtered)

# Combine all lyrics, excluding words from the exclude list
filtered_lyrics = ' '.join(
    data['cleaned_lyrics'].apply(lambda x: remove_excluded_words(x, exclude_list))
)

# Generate the WordCloud with filtered lyrics
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_lyrics)

# Plot the WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Most Common Words in Song Lyrics (Filtered)")
plt.show()
