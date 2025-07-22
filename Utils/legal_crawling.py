import sys
import time
import requests
from urllib.parse import urljoin, urldefrag, urlparse, urlunparse 
import nltk
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
from nltk.corpus import stopwords
import pickle
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import heapq 
import urllib.robotparser 
import hashlib
import logging
import signal
from Utils.text_preprocessor import preprocess_text


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("log/crawler.log", mode='a', encoding='utf-8'),  # Save logs to file (overwrite)
        logging.StreamHandler()  # Print logs to console
    ]
)

class OfflineCrawler:
    def __init__(self, seeds, max_depth=2, delay=0.5, simhash_threshold=3):
        self.seeds = seeds
        self.max_depth = max_depth
        self.default_delay = delay # Store the default delay separately
        self.simhash_threshold = simhash_threshold

        self.path_to_crawled_data = 'data/crawled_data.pkl'
        self.path_to_simhashes = 'data/simhashes.pkl'
        self.path_to_frontier = 'data/frontier.pkl'
        self.path_to_visited_urls_in_queue = 'data/visited_urls_in_queue.pkl'

        self.crawled_data = self._load(self.path_to_crawled_data)
        self.seen_simhashes = self._load(self.path_to_simhashes)
        self.frontier = self._load(self.path_to_frontier) 
        self.visited_urls_in_queue = self._load(self.path_to_visited_urls_in_queue) 

        # some ways to save frontier and visited URLs during crawl
        signal.signal(signal.SIGINT, self._handle_interrupt) 
        self.crawled_since_last_save = 0
        self.save_threshold = 10 # Save every 100 crawled documents

        # For mapping URL to doc_id (helpful for debugging and future relevance)
        self.url_to_doc_id = {url_info['url']: doc_id for doc_id, url_info in self.crawled_data.items() if 'url' in url_info}

        # Stores RobotFileParser objects, keyed by domain (netloc)
        self.robot_parsers = {}
        # Stores the effective crawl delay for each domain
        self.domain_delays = {}
        # User-Agent name
        self.user_agent = "TübingenSearchBot_UniProject/4.20 (https://alma.uni-tuebingen.de/alma/pages/startFlow.xhtml?_flowId=detailView-flow&unitId=78284&periodId=228&navigationPosition=studiesOffered,searchCourses)" 
            
        #Tiered keywords for weighted priority
        self.very_relevant_keywords = {
            "t%C3%BCbingen","T%C3%BCbingen",
            "tuebingen", "tubingen",
            "eberhard karls universität", "hohentuebingen", "hohentübingen", "schloss hohentübingen",
            "stocherkahnrennen", "chocolart",
            "university", "universität", # Keep both for breadth
            "neckar", "altstadt", # Altstadt is commonly used in English context
            "baden-wuerttemberg", "baden-württemberg",
            "hölderlinturm", "hölderlin tower", # Keep both
            "neckarfront", # Commonly used in English context
            "stiftskirche", "collegiate church", # Keep both
            "marktplatz", "market square", # Keep both
            "rathaus", "town hall", # Keep both
            "museum", "mut", # MUT is specific
            "botanical garden", "botanischer garten", # Keep both
            "fachwerkhaus", "half-timbered house" # Keep both
        }
        self.relevant_keywords = {
            "city", "stadtfest", # Stadtfest is common
            "cyber valley", "uniklinik", "university hospital", # Keep both
            "tourism", "visitors",
            "restaurants", "cafes", "gastronomy", "food", "drinks", "bars", "bakery",
            "butcher", "market", "wochenmarkt", # Wochenmarkt is common
            "schwaebische alb", "schwübische alb", "schoenbuch", "schönbuch", # Regional names
            "umbrisch-provenzalischer markt", "jazz & klassik tage", "sommernachtskino", # Specific event names
            "weihnachtsmarkt", "christmas market", # Keep both
            "filmfest", "filmfestival", "umweltfilmfestival", # Keep both filmfest/festival, umwelt specific
            "literaturfestival", "literature festival", # Keep both
            "tübingen rennt", "stadtlauf", # Specific event names
            "maultaschen", "spätzle", # Food names
            "wein", "wine", # Keep both
            "brauerei", "brewery", # Keep both
            "biergarten", "beer garden", # Keep both
            "stocherkahn", "punt boat", "punting", # Keep all
            "students", "student life", "nightlife",
            "max planck institute", "hertie institute", # Specific institute names
            "ki", "ai", "artificial intelligence", # Common acronyms/terms
            "biotechnology", "neuroscience" # Research fields
        }
        self.moderately_relevant_keywords = {
            "community", "information", "news", "events", "travel", "sights", "attractions", "living",
            "citizens", "local", "region", "research", "science", "study", "students",
            "reutlingen", "rottenburg am neckar", "moessingen", "mössingen", # Neighboring towns
            "kirchentellinsfurt", "ammerbuch", "dusslingen", "dußlingen", "gomaringen",
            "kusterdingen", "ofterdingen", "nehren", "dettenhausen", # Neighboring towns
            "housing", "real estate", "traffic", "public transport", "parking",
            "health", "doctors", "pharmacies", "schools", "kindergartens", "sport",
            "clubs", "associations", "town hall", "administration", "waste", "building",
            "culture", "art", "theater", "theatre", "museums", "exhibitions", "music",
            "cinema", "books", "literature", "galleries",
            "naturpark schönbuch", "schönbuch nature park", # Keep both
            "neckartal", "neckar valley", # Keep both
            "hiking", "wandern", # Keep both
            "cycling", "radfahren", # Keep both
            "castle", "burg", # Keep both
            "bus", "train", "deutsche bahn", # Deutsche Bahn is specific
            "bahnhof", "train station", # Keep both
            "bürgerbüro", "citizen's office", # Keep both
            "wirtschaftsförderung", "economic development", # Keep both
            "integration", # Relevant
            "volkshochschule", "adult education center", # Keep both
            "kunsthalle", "landestheater", # Specific institutions
            "galerie", "gallery" # Keep both
        }
        
        # NEW: Tiered Domain Lists
        self.high_prio_domains = {
            "tuebingen.de", 
            "uni-tuebingen.de", 
            "visit-tuebingen.info", # Often official tourism site
            "stadtmuseum-tuebingen.de",
            "schloss-hohentuebingen.de", # Example official site
            "stiftskirche-tuebingen.de", # Example official site
            # Add other official Tübingen-related sites here
        }

        # Domains to be put last in queue (effectively, very low priority)
        # These are domains that are known to be problematic, mostly irrelevant, or overwhelming.
        self.blacklisted_domains = {
            "pinterest.com", # Often leads to images/social media, not rich content
            "facebook.com", "instagram.com", "twitter.com", "youtube.com", # Social media
            "maps.google.com", "google.com", # Search/Map related
            "amazon.com", "ebay.com", # E-commerce
            "booking.com", "getyourguide.com", "viator.com", # General booking sites (unless specifically target relevant subpaths)
            # You can add the problematic "speisekartenweb.de" here if it's always German and irrelevant
            "speisekartenweb.de"
        }
        # The remaining domains will be considered "normal"

        # NEW: URL Ending Blacklist (Top-level domains or file extensions)
        self.url_ending_blacklist = {
            ".ru", ".cn", ".br", ".fr", ".es", ".it", ".jp", ".kr", ".pt", ".cz", ".pl", ".ch", # Common non-English TLDs
            ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".zip", ".rar", # Document/Archive files
            ".jpg", ".jpeg", ".png", ".gif", ".svg", ".mp4", ".mov", ".avi", ".mp3", ".wav", # Media files
            ".exe", ".dmg", ".apk", ".iso", # Executables
            ".xml", ".rss", ".atom", # Feeds/XML (unless specifically targeting sitemaps for links)
            ".css", ".js", # Styling/Scripting files
            "tel:", "mailto:", "ftp:", # Non-HTTP/HTTPS schemes
            "javascript:", "#" # In-page anchors or javascript links
        }

        self.blacklisted_city_keywords = [
            # Existing entries
            "paris", "london", "rome", "new york", "tokyo", "dubai", "barcelona",
            "amsterdam", "sydney", "hong kong", "bangkok", "istanbul", "los angeles",
            "las vegas", "miami", "singapore", "kuala lumpur", "prague", "berlin",
            "vienna", "madrid", "athens", "venice", "florence", "munich",
            "copenhagen", "stockholm", "oslo", "helsinki", "dublin", "edinburgh",
            "montreal", "toronto", "vancouver", "rio de janeiro", "buenos aires",
            "santiago", "lima", "mexico city", "beijing", "shanghai", "delhi",
            "mumbai", "seoul", "kyoto", "osaka", "hanoi", "ho chi minh city",
            "melbourne", "auckland", "cairo", "marrakech", "cape town", "jerusalem",
            "tel aviv",
            
            # German cities (excluding Tübingen region)
            "hamburg", "frankfurt", "dresden", "leipzig", "nuremberg", "nürnberg",
            "köln", "cologne", "bremen", "hannover", "duisburg", "dortmund",
            "essen", "bochum", "bonn", "wiesbaden", "mainz", "rostock",
            "kiel", "magdeburg", "potsdam", "erfurt", "regensburg", "augsburg",
            "freiburg", "saarbrücken", "karlsruhe", "mannheim", "heidelberg", "trier",
            "lübeck", "coburg", "weimar", "bamberg", "würzburg", "passau",
            "konstanz", "lindau", "garmisch", "berchtesgaden", "rothenburg",
            "dinkelsbühl", "bad reichenhall", "aachen", "münster", "bielefeld",
            "wolfsburg", "braunschweig", "göttingen", "kassel", "fulda",
            "marburg", "giessen", "darmstadt", "offenbach", "worms", "speyer",
            "landau", "neustadt", "baden-baden", "pforzheim", "heilbronn",
            "schwäbisch hall", "ulm", "ravensburg", "friedrichshafen", "bodensee",
            "memmingen", "kempten", "füssen", "oberammergau", "mittenwald",
            
            # Austria
            "vienna", "wien", "salzburg", "innsbruck", "graz", "linz", "klagenfurt",
            "villach", "wels", "steyr", "dornbirn", "feldkirch", "bregenz",
            "eisenstadt", "st. pölten", "krems", "melk", "hallstatt", "st. anton",
            "zell am see", "kitzbühel", "sölden", "kaprun", "bad gastein",
            "schladming", "saalbach", "mayrhofen", "zillertal",
            
            # Switzerland
            "zurich", "geneva", "basel", "bern", "lausanne", "winterthur", "lucerne",
            "st. gallen", "lugano", "biel", "thun", "köniz", "la chaux-de-fonds",
            "schaffhausen", "fribourg", "vernier", "chur", "neuchâtel", "uster",
            "sion", "interlaken", "zermatt", "st. moritz", "davos", "grindelwald",
            "wengen", "mürren", "saas-fee", "verbier", "crans-montana", "gstaad",
            "andermatt", "engelberg", "flims", "arosa", "locarno", "ascona",
            "montreux", "vevey", "nyon", "morges", "yverdon", "bellinzona",
            
            # France
            "nice", "marseille", "lyon", "bordeaux", "toulouse", "lille", "nantes",
            "strasbourg", "montpellier", "rennes", "reims", "toulon", "saint-étienne",
            "dijon", "angers", "grenoble", "villeurbanne", "aix-en-provence",
            "brest", "limoges", "tours", "amiens", "perpignan", "metz", "besançon",
            "orléans", "mulhouse", "caen", "boulogne-billancourt", "rouen", "nancy",
            "argenteuil", "roubaix", "tourcoing", "nîmes", "avignon", "poitiers",
            "fort-de-france", "versailles", "créteil", "pau", "vitry-sur-seine",
            "cannes", "antibes", "saint-tropez", "biarritz", "deauville", "chamonix",
            "megève", "courchevel", "val d'isère", "tignes", "les deux alpes",
            "alpe d'huez", "la plagne", "val thorens", "méribel", "annecy",
            "chambéry", "aix-les-bains", "évian", "thonon", "albertville",
            "carcassonne", "mont-saint-michel", "saint-malo", "quimper", "vannes",
            
            # Italy
            "milano", "milan", "naples", "palermo", "turin", "genoa", "bologna",
            "catania", "bari", "messina", "verona", "trieste", "padua", "brescia",
            "taranto", "prato", "parma", "modena", "reggio calabria", "reggio emilia",
            "perugia", "ravenna", "livorno", "cagliari", "foggia", "rimini", "salerno",
            "ferrara", "sassari", "latina", "giugliano", "monza", "syracuse", "bergamo",
            "pescara", "trento", "vicenza", "terni", "forlì", "bolzano", "novara",
            "piacenza", "ancona", "andria", "arezzo", "udine", "cesena", "lecce",
            "pisa", "siena", "assisi", "san gimignano", "lucca", "pistoia",
            "montepulciano", "cortona", "chianti", "cinque terre", "portofino",
            "bellagio", "como", "stresa", "sorrento", "positano", "amalfi",
            "capri", "taormina", "cefalù", "alghero", "costa smeralda",
            
            # Spain
            "madrid", "barcelona", "valencia", "seville", "zaragoza", "málaga",
            "murcia", "palma", "las palmas", "bilbao", "alicante", "córdoba",
            "valladolid", "vigo", "gijón", "hospitalet", "vitoria", "coruña",
            "granada", "elche", "oviedo", "badalona", "cartagena", "terrassa",
            "jerez", "sabadell", "móstoles", "santa cruz", "pamplona", "almería",
            "fuenlabrada", "leganés", "donostia", "san sebastián", "burgos",
            "santander", "castellón", "getafe", "alcorcón", "logroño", "badajoz",
            "salamanca", "huelva", "marbella", "lleida", "tarragona", "león",
            "cádiz", "dos hermanas", "estepa", "parla", "torrejón", "reus",
            "toledo", "segovia", "ávila", "cáceres", "cuenca", "santiago",
            "ronda", "mérida", "úbeda", "baeza", "chinchón", "aranjuez",
            
            # Netherlands
            "amsterdam", "rotterdam", "the hague", "utrecht", "eindhoven", "tilburg",
            "groningen", "almere", "breda", "nijmegen", "enschede", "haarlem",
            "arnhem", "zaanstad", "s-hertogenbosch", "amersfoort", "maastricht",
            "dordrecht", "leiden", "haarlemmermeer", "zoetermeer", "zwolle",
            "delft", "alkmaar", "leeuwarden", "volendam", "marken", "kinderdijk",
            "keukenhof", "giethoorn", "zaanse schans", "edam", "gouda", "naarden",
            
            # Belgium
            "brussels", "antwerp", "ghent", "charleroi", "liège", "bruges", "namur",
            "leuven", "mons", "aalst", "mechelen", "la louvière", "kortrijk",
            "hasselt", "sint-niklaas", "ostend", "tournai", "genk", "seraing",
            "roeselare", "verviers", "mouscron", "beveren", "dendermonde",
            "beringen", "turnhout", "vilvoorde", "lokeren", "sint-truiden",
            "waterloo", "dinant", "durbuy", "spa", "knokke", "de haan",
            
            # Portugal
            "lisbon", "porto", "amadora", "braga", "setúbal", "coimbra", "queluz",
            "funchal", "cacém", "vila nova de gaia", "loures", "felgueiras",
            "évora", "rio maior", "odivelas", "aveiro", "corroios", "barreiro",
            "sintra", "cascais", "óbidos", "monsaraz", "porto de mós", "nazaré",
            "fátima", "batalha", "alcobaça", "tomar", "guimarães", "viana do castelo",
            "lagos", "tavira", "sagres", "albufeira", "portimão", "faro",
            
            # Eastern Europe
            "warsaw", "krakow", "gdansk", "wroclaw", "poznan", "lodz", "szczecin",
            "bydgoszcz", "lublin", "katowice", "białystok", "gdynia", "częstochowa",
            "radom", "sosnowiec", "torun", "kielce", "gliwice", "zabrze", "bytom",
            "olsztyn", "bielsko-biała", "rzeszów", "ruda śląska", "rybnik",
            "tychy", "dąbrowa górnicza", "płock", "elbląg", "opole", "gorzów",
            "wałbrzych", "włocławek", "tarnów", "chorzów", "koszalin", "kalisz",
            "legnica", "grudziądz", "słupsk", "jaworzno", "jastrzębie",
            "zakopane", "wieliczka", "malbork", "auschwitz",
            
            # Czech Republic
            "prague", "brno", "ostrava", "plzen", "liberec", "olomouc", "budejovice",
            "hradec kralove", "usti nad labem", "pardubice", "zlin", "havirov",
            "kladno", "most", "opava", "frydek-mistek", "karviná", "jihlava",
            "teplice", "decin", "karlovy vary", "jablonec", "mlada boleslav",
            "prostejov", "prerov", "ceske budejovice", "trebic", "tabor",
            "krumlov", "telc", "kutna hora", "marianske lazne", "frantiskovy lazne",
            
            # Nordic countries
            "stockholm", "göteborg", "malmö", "uppsala", "västerås", "örebro",
            "linköping", "helsingborg", "jönköping", "norrköping", "lund", "umeå",
            "gävle", "borås", "eskilstuna", "halmstad", "växjö", "karlstad",
            "sundsvall", "trollhättan", "östersund", "falun", "kalmar", "kristianstad",
            "visby", "kiruna", "abisko", "are", "götland", "öland",
            
            "copenhagen", "aarhus", "odense", "aalborg", "esbjerg", "randers",
            "kolding", "horsens", "vejle", "roskilde", "herning", "hørsholm",
            "helsingør", "silkeborg", "næstved", "fredericia", "viborg", "køge",
            "holstebro", "taastrup", "slagelse", "hillerød", "sønderborg",
            
            "oslo", "bergen", "stavanger", "trondheim", "fredrikstad", "drammen",
            "skien", "kristiansand", "ålesund", "tønsberg", "moss", "haugesund",
            "sandefjord", "bodø", "tromsø", "harstad", "porsgrunn", "arendal",
            "hamar", "ytrebygda", "larvik", "halden", "askøy", "molde",
            "flåm", "geiranger", "lofoten", "nordkapp", "alta", "hammerfest",
            
            "helsinki", "espoo", "tampere", "vantaa", "oulu", "turku", "jyväskylä",
            "lahti", "kuopio", "pori", "joensuu", "lappeenranta", "hämeenlinna",
            "vaasa", "seinäjoki", "rovaniemi", "mikkeli", "kotka", "salo",
            "porvoo", "kouvola", "hyvinkää", "järvenpää", "rauma", "tuusula",
            "kirkkonummi", "kajaani", "kerava", "naantali", "savonlinna",
            "kemi", "tornio", "inari", "kittilä",
            
            # Other European destinations
            "reykjavik", "akureyri", "keflavik", "vik", "höfn", "selfoss",
            "husavik", "egilsstadir", "isafjordur", "westfjords", "blue lagoon",
            
            "valletta", "mdina", "sliema", "st. julian's", "bugibba", "marsaxlokk",
            "gozo", "comino", "rabat", "birgu", "cospicua", "senglea",
            
            "luxembourg", "esch-sur-alzette", "differdange", "dudelange",
            "ettelbruck", "diekirch", "wiltz", "echternach", "grevenmacher",
            "remich", "vianden", "clervaux", "beaufort", "bourscheid",
            
            "andorra la vella", "escaldes-engordany", "encamp", "sant julià",
            "la massana", "ordino", "canillo", "soldeu", "pas de la casa",
            
            "monaco", "monte carlo", "fontvieille", "la condamine", "moneghetti",
            
            "san marino", "borgo maggiore", "serravalle", "domagnano",
            "fiorentino", "acquaviva", "faetano", "montegiardino", "chiesanuova",
            
            "vatican city", "st. peter's", "sistine chapel",
            
            # Balkan region
            "zagreb", "split", "rijeka", "osijek", "zadar", "pula", "sesvete",
            "slavonski brod", "karlovac", "varazdin", "sisak", "velika gorica",
            "dubrovnik", "trogir", "rovinj", "motovun", "plitvice", "krka",
            "hvar", "korcula", "brac", "vis", "mljet", "paklenica",
            
            "ljubljana", "maribor", "celje", "kranj", "velenje", "nova gorica",
            "koper", "novo mesto", "ptuj", "trbovlje", "kamnik", "jesenice",
            "bled", "bohinj", "portoroz", "piran", "skocjan", "postojna",
            "vipava", "lipica", "rogaska slatina", "lasko", "terme catez",
            
            "belgrade", "novi sad", "nis", "kragujevac", "subotica", "pancevo",
            "cacak", "novi pazar", "zrenjanin", "leskovac", "smederevo", "valjevo",
            "kruševac", "vranje", "sombor", "pozarevac", "pirot", "zajecar",
            "kikinda", "sremska mitrovica", "jagodina", "vršac", "bor",
            
            "sarajevo", "banja luka", "tuzla", "zenica", "mostar", "bijeljina",
            "brčko", "bihać", "prijedor", "trebinje", "doboj", "cazin",
            "velika kladuša", "visoko", "goražde", "konjic", "travnik",
            "jajce", "livno", "široki brijeg", "gradačac", "bugojno",
            
            "skopje", "bitola", "kumanovo", "prilep", "tetovo", "veles", "stip",
            "ohrid", "gostivar", "strumica", "kavadartsi", "gevgelija", "kocani",
            "radovis", "vinica", "delcevo", "kriva palanka", "sveti nikole",
            "negotino", "krusevo", "struga", "debar", "makedonski brod",
            
            "tirana", "durrës", "vlorë", "elbasan", "shkodër", "fier", "korçë",
            "berat", "lushnjë", "kavajë", "gjirokastër", "sarandë", "kukes",
            "lezhë", "pogradec", "peshkopi", "burrel", "corovode", "përmet",
            "butrint", "blue eye", "theth", "valbona", "ksamil", "himara",
            
            "podgorica", "nikšić", "pljevlja", "bijelo polje", "cetinje", "bar",
            "herceg novi", "berane", "budva", "ulcinj", "tivat", "rozaje",
            "plav", "zabljak", "andrijevica", "danilovgrad", "kolasin",
            "perast", "kotor", "sveti stefan", "durmitor", "biogradska gora",
            
            "sofia", "plovdiv", "varna", "burgas", "ruse", "stara zagora",
            "pleven", "sliven", "dobrich", "shumen", "pernik", "haskovo",
            "yambol", "pazardzhik", "blagoevgrad", "veliko tarnovo", "vratsa",
            "gabrovo", "asenovgrad", "vidin", "kazanlak", "kyustendil",
            "bansko", "borovets", "pamporovo", "sozopol", "nessebar", "albena",
            "golden sands", "sunny beach", "primorsko", "tsarevo", "sinemorets",
            
            "bucharest", "cluj-napoca", "timisoara", "iasi", "constanta", "craiova",
            "brasov", "galati", "ploiesti", "oradea", "braila", "arad", "pitesti",
            "sibiu", "bacau", "targu mures", "baia mare", "buzau", "botosani",
            "satu mare", "ramnicu valcea", "drobeta-turnu severin", "suceava",
            "piatra neamt", "targu jiu", "tulcea", "focsani", "bistrita",
            "sighisoara", "sinaia", "brasov", "bran", "rasnov", "peles",
            "dracula", "transylvania", "maramures", "bucovina", "danube delta",
            
            "budapest", "debrecen", "szeged", "miskolc", "pécs", "győr", "nyíregyháza",
            "kecskemét", "székesfehérvár", "szombathely", "szolnok", "tatabánya",
            "kaposvár", "békéscsaba", "érd", "zalaegerszeg", "sopron", "eger",
            "nagykanizsa", "dunakeszi", "hódmezővásárhely", "cegléd", "veszprém",
            "lake balaton", "szentendre", "visegrád", "esztergom", "hévíz",
            "tokaj", "aggtelek", "hortobágy", "pécs", "fertő", "hollókő",
            
            "bratislava", "košice", "prešov", "žilina", "nitra", "banská bystrica",
            "trnava", "martin", "trenčín", "poprad", "prievidza", "zvolen",
            "považská bystrica", "nové zámky", "michalovce", "spišská nová ves",
            "komárno", "levice", "bardejov", "liptovský mikuláš", "ružomberok",
            "high tatras", "spiš castle", "bojnice", "orava", "danube",
            
            "vilnius", "kaunas", "klaipeda", "šiauliai", "panevėžys", "alytus",
            "marijampolė", "mažeikiai", "jonava", "utena", "kėdainiai", "telšiai",
            "visaginas", "tauragė", "ukmergė", "plungė", "kretinga", "šilutė",
            "trakai", "palanga", "nida", "druskininkai", "birštonas", "anykščiai",
            
            "riga", "daugavpils", "liepaja", "jelgava", "jūrmala", "ventspils",
            "rēzekne", "valmiera", "jēkabpils", "ogre", "tukums", "salaspils",
            "cēsis", "kuldīga", "saldus", "talsi", "dobele", "krāslava",
            "sigulda", "bauska", "rundāle", "gauja", "kemeri", "cape kolka",
            
            "tallinn", "tartu", "narva", "pärnu", "kohtla-järve", "viljandi",
            "rakvere", "maardu", "sillamäe", "kuressaare", "võru", "valga",
            "haapsalu", "jõhvi", "keila", "paide", "kiviõli", "tapa",
            "saaremaa", "hiiumaa", "lahemaa", "soomaa", "matsalu", "vilsandi",
            
            # Others from around the world often mentioned on travel sites
            "san francisco", "chicago", "washington", "orlando", "boston",
            "seattle", "honolulu", "moscow", "saint petersburg", "dubai", "doha",
            "abu dhabi", "manila", "jakarta", "tehran", "bali", "phuket",
            "patong", "koh samui", "koh phi phi", "krabi", "pattaya", "hua hin",
            "chiang mai", "chiang rai", "pai", "sukhothai", "ayutthaya",
            "kanchanaburi", "lopburi", "nakhon ratchasima", "ubon ratchathani",
            "khon kaen", "nong khai", "mae hong son", "phitsanulok", "lampang"
        ]


        # Add initial seeds to the frontier (ensures they are high priority at start/resume)
        self._add_initial_seeds(seeds)

    def run(self):
        self.last_fetch_time = {}

        seeds_already_known = 0
        seeds_added_count = 0

        # Process initial seeds (can be a subset of self.seeds) to populate frontier
        # This loop only runs once at the beginning of the crawl
        for url in self.seeds:
            # MODIFIED: Normalize URL before checks for consistency
            normalized_url, _ = urldefrag(urljoin(url, url))
            
            # --- NEW/MODIFIED: Robust URL parsing for seeds ---
            try:
                parsed_url = urlparse(normalized_url)
            except Exception as e:
                logging.warning(f"[SEED_SKIP] Error parsing seed URL '{url}': {e}. Skipping.")
                seeds_already_known += 1
                continue

            # Skip malformed URLs early
            if not parsed_url.scheme or not parsed_url.netloc:
                logging.warning(f"[SEED_SKIP] Malformed seed URL: '{url}'. Skipping.")
                seeds_already_known += 1
                continue

            # Check if already processed (crawled and indexed)
            if normalized_url in self.url_to_doc_id:
                logging.info(f"[SEED_SKIP] Seed '{normalized_url}' already processed (Doc ID: {self.url_to_doc_id[normalized_url]}).")
                seeds_already_known += 1
                continue

            # Check if already in frontier or visited_urls_in_queue
            if normalized_url in self.visited_urls_in_queue:
                logging.info(f"[SEED_SKIP] Seed '{normalized_url}' already in frontier or visited. Skipping.")
                seeds_already_known += 1
                continue

            # NEW: Check for blacklisted domains or URL endings for seeds
            if parsed_url.netloc in self.blacklisted_domains or self._is_blacklisted_ending(normalized_url):
                logging.warning(f"[SEED_SKIP] Seed '{normalized_url}' domain/ending blacklisted. Skipping.")
                seeds_already_known += 1
                continue

            # Check robots.txt for the seed URL before adding
            rp = self._get_robot_parser(normalized_url)
            if rp.can_fetch(self.user_agent, normalized_url):
                priority = self._calculate_priority(normalized_url, "", [], 0)
                heapq.heappush(self.frontier, (priority, normalized_url, 0))
                self.visited_urls_in_queue.add(normalized_url) # Add to visited_urls_in_queue when pushed to frontier
                logging.info(f"[SEED_ADD] Added new seed to frontier: '{normalized_url}'.")
                seeds_added_count += 1
            else:
                logging.info(f"[SEED_SKIP] Seed '{normalized_url}' disallowed by robots.txt.")
                seeds_already_known += 1

        logging.info(f"[START CRAWL] Frontier initialized with {len(self.frontier)} URLs.")
        logging.info(f"Max depth: {self.max_depth}, Default delay: {self.default_delay}s, SimHash threshold: {self.simhash_threshold}")

        # Track some statistics
        crawled_count = 0
        skipped_robots = 0
        skipped_duplicate_content = 0
        skipped_non_english = 0
        skipped_already_processed = 0
        skipped_blacklisted = 0
        skipped_fetch_errors = 0 
        skipped_parsing_errors = 0 

        while self.frontier:
            priority, url, depth = heapq.heappop(self.frontier)

            # --- Initial URL validation and Filtering (Most Efficient) ---
            # NEW/MODIFIED: Robust URL parsing for URLs from frontier
            try:
                parsed_url = urlparse(url)
                if not parsed_url.scheme or not parsed_url.netloc:
                    logging.warning(f"[MALFORMED_URL] Skipping malformed URL from frontier: '{url}'.")
                    skipped_fetch_errors += 1 # Count as a fetch error (can't even form a request)
                    continue
                domain = parsed_url.netloc
            except Exception as e:
                logging.error(f"[URL_PARSE_ERROR] Failed to parse URL '{url}': {e}. Skipping.")
                skipped_fetch_errors += 1
                continue

            if self._is_blacklisted_ending(url):
                logging.info(f"[BLACKLIST_ENDING] Skipping {url} due to blacklisted URL ending.")
                skipped_blacklisted += 1
                continue

            if domain in self.blacklisted_domains:
                logging.info(f"[BLACKLIST_DOMAIN] Skipping {url} due to blacklisted domain: {domain}.")
                skipped_blacklisted += 1
                continue
            # --- END Initial Filtering ---


            # Check if this URL has been *processed* before (based on doc_id)
            if url in self.url_to_doc_id:
                logging.info(f"[ALREADY_PROCESSED] Skipping {url} (already processed and indexed in a previous run).")
                skipped_already_processed += 1
                continue

            if depth > self.max_depth:
                logging.info(f"Skipping {url} (max depth {self.max_depth} reached).")
                continue

            rp = self._get_robot_parser(url)
            if not rp.can_fetch(self.user_agent, url):
                logging.info(f"[ROBOTS] Skipping {url} due to robots.txt rules for {domain}.")
                skipped_robots += 1
                continue

            required_delay = self.domain_delays.get(domain, self.default_delay)

            if domain in self.last_fetch_time:
                time_since_last_fetch = time.time() - self.last_fetch_time[domain]
                if time_since_last_fetch < required_delay:
                    sleep_duration = required_delay - time_since_last_fetch
                    logging.info(f"[DELAY] Sleeping for {sleep_duration:.2f}s for {domain} to respect crawl-delay.")
                    time.sleep(sleep_duration)

            self.last_fetch_time[domain] = time.time()

            start_page_time = time.time()

            logging.info(f"\n[CRAWL] Fetching: {url} (Depth: {depth}, Priority: {-priority:.2f})")
            html = None # Initialize html to None for clear scope

            # --- NEW/MODIFIED: REQUESTS ERROR CATCHING & ENCODING HANDLING ---
            try:
                headers = {'User-Agent': self.user_agent}
                # Using stream=True for potentially problematic responses, and explicit decode
                # to handle encoding errors gracefully instead of crashing.
                with requests.get(url, headers=headers, timeout=15, stream=True) as resp: # Increased timeout slightly
                    resp.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                    # Attempt to read content with resp.encoding, fall back to utf-8, then iso-8859-1
                    # This is the most robust way to handle diverse encodings.
                    try:
                        html = resp.content.decode(resp.encoding if resp.encoding else 'utf-8')
                    except UnicodeDecodeError:
                        logging.warning(f"[ENCODING_ERROR] UnicodeDecodeError for {url} with detected encoding '{resp.encoding}'. Trying utf-8 then iso-8859-1.")
                        try:
                            html = resp.content.decode('utf-8', errors='replace') # Replace invalid chars
                        except UnicodeDecodeError:
                            logging.warning(f"[ENCODING_ERROR] Failed to decode {url} with utf-8. Trying iso-8859-1.")
                            html = resp.content.decode('iso-8859-1', errors='replace') # Last resort
                            
            except requests.exceptions.RequestException as e:
                # Catches ConnectionError, Timeout, HTTPError (4xx, 5xx), TooManyRedirects, etc.
                logging.error(f"[FETCH_ERROR] Error fetching {url}: {e}. Skipping this URL.")
                skipped_fetch_errors += 1
                continue # Skip to the next URL in the frontier

            # Check if HTML content was actually obtained after decoding attempts
            if not html:
                logging.error(f"[FETCH_ERROR] No HTML content obtained for {url} after decoding attempts. Skipping.")
                skipped_fetch_errors += 1
                continue

            # --- NEW/MODIFIED: BEAUTIFULSOUP PARSING ERROR CATCHING ---
            soup = None
            try:
                soup = BeautifulSoup(html, "html.parser")
                # A very basic sanity check: if no <body> tag is found, it might be truly malformed.
                if not soup.body:
                    logging.warning(f"[PARSING_WARN] Page {url} appears to have no body content after parsing. Skipping.")
                    skipped_parsing_errors += 1
                    continue
            except Exception as e:
                # This will catch bs4.exceptions.ParserRejectedMarkup and other parsing errors
                logging.error(f"[PARSING_ERROR] Error parsing HTML for {url}: {e}. Skipping this URL.")
                skipped_parsing_errors += 1
                continue # Skip to the next URL in the frontier


            # --- SIMHASH AND CONTENT PROCESSING (unchanged core logic, but now protected by try-excepts above) ---
            cleaned_text_for_simhash = self._get_cleaned_text_for_simhash(soup)
            if cleaned_text_for_simhash:
                current_page_simhash = self._compute_simhash(cleaned_text_for_simhash)
            else:
                # If no text is extracted for simhash, treat as potential duplicate or non-content page
                logging.info(f"[SIMHASH_WARN] No significant text for simhash from {url}. Skipping to avoid processing empty content.")
                skipped_duplicate_content += 1 # Count as a type of content skip
                continue

            is_content_duplicate = False
            for existing_simhash in self.seen_simhashes:
                if self._hamming_distance(current_page_simhash, existing_simhash) <= self.simhash_threshold:
                    logging.info(f"[SIMHASH] Skipping {url} (content duplicate with existing hash).")
                    is_content_duplicate = True
                    skipped_duplicate_content += 1
                    break

            if is_content_duplicate:
                continue

            self.seen_simhashes.add(current_page_simhash)
            self._save(self.path_to_simhashes, self.seen_simhashes)

            # Preprocess text for indexing
            tokens_for_indexing = preprocess_text(soup) # Assuming preprocess_text is robust
            if not tokens_for_indexing:
                logging.info(f"[LANG] Skipping {url} (non-English or no extractable text after preprocessing).")
                skipped_non_english += 1
                continue

            doc_id = self._get_id(url)
            self._save_crawled(doc_id, {'url': url, 'tokens': tokens_for_indexing})
            self.url_to_doc_id[url] = doc_id
            crawled_count += 1

            # NEW: Incremented counter for periodic saving
            self.crawled_since_last_save += 1

            # NEW: Periodic Saving Check - Place this after a page is successfully crawled and all its state is updated
            if self.crawled_since_last_save >= self.save_threshold:
                logging.info(f"[PERIODIC_SAVE] Performing periodic save after {self.crawled_since_last_save} successful crawls.")
                try:
                    self._save(self.path_to_frontier, self.frontier)
                    self._save(self.path_to_visited_urls_in_queue, self.visited_urls_in_queue)
                    self._save(self.path_to_simhashes, self.seen_simhashes) # Ensure simhashes are also saved here
                    # self._save(self.path_to_crawled_data, self.crawled_data) # crawled_data is already saved more often, but no harm in saving again.
                    # Or, better, if you implement _save_crawled to save all data types, you can just call it.
                except Exception as e:
                    logging.error(f"[PERIODIC_SAVE_ERROR] Failed during periodic save: {e}")
                finally:
                    self.crawled_since_last_save = 0 # Reset counter after attempting to save

            end_page_time = time.time()
            processing_time = end_page_time - start_page_time
            logging.info(f"[SUCCESS] Processed {url} (Doc ID: {doc_id}) in {processing_time:.2f} seconds.")

            # --- Link Extraction and Frontier Management ---
            if depth < self.max_depth:
                current_page_text_for_scoring = self._get_cleaned_text_for_simhash(soup)
                current_page_tokens_for_scoring = self._clean_for_scoring(current_page_text_for_scoring)

                links_added_from_page = 0
                for a in soup.find_all("a", href=True):
                    href = urljoin(url, a["href"])
                    href, _ = urldefrag(href)
                    
                    # --- NEW/MODIFIED: Robust parsing of extracted href ---
                    try:
                        p = urlparse(href)
                    except Exception as e:
                        logging.warning(f"[LINK_PARSE_ERROR] Skipping malformed extracted link '{href}' from '{url}': {e}.")
                        continue

                    if p.scheme in ("http", "https"):
                        # --- NEW: Immediate Filtering for new links ---
                        if self._is_blacklisted_ending(href):
                            continue # Skip adding this link
                        if p.netloc in self.blacklisted_domains:
                            continue # Skip adding this link
                        # --- END Immediate Filtering ---

                        # NEW: Check if already processed (crawled and indexed) or already in frontier
                        if href in self.url_to_doc_id or href in self.visited_urls_in_queue:
                            continue

                        target_rp = self._get_robot_parser(href)
                        if target_rp.can_fetch(self.user_agent, href):
                            anchor_text = a.get_text(strip=True)

                            new_priority = self._calculate_priority(
                                href,
                                anchor_text,
                                current_page_tokens_for_scoring,
                                depth + 1
                            )

                            if new_priority != -float('inf'): # Only add if not effectively blacklisted by _calculate_priority
                                heapq.heappush(self.frontier, (new_priority, href, depth + 1))
                                self.visited_urls_in_queue.add(href)
                                links_added_from_page += 1
                            else:
                                pass # Logging for blacklisted URLs already done by _calculate_priority
                        else:
                            pass # Logging for robots.txt disallow can be very verbose for all links

                logging.info(f"[LINKS] Added {links_added_from_page} new links to frontier. Frontier size: {len(self.frontier)}.")
            else:
                logging.info(f"[LINKS] Max depth reached, no new links added from {url}.")

        logging.info(f"\n[CRAWL COMPLETE] Finished crawling.")
        logging.info(f"Total pages crawled and indexed: {crawled_count}")
        logging.info(f"Skipped by robots.txt: {skipped_robots}")
        logging.info(f"Skipped as content duplicates (SimHash): {skipped_duplicate_content}")
        logging.info(f"Skipped as non-English or no text: {skipped_non_english}")
        logging.info(f"Skipped (already processed in previous run): {skipped_already_processed}")
        logging.info(f"Skipped (blacklisted domain/ending): {skipped_blacklisted}")
        logging.info(f"Skipped due to fetch/request errors: {skipped_fetch_errors}") # NEW stat
        logging.info(f"Skipped due to HTML parsing errors: {skipped_parsing_errors}") # NEW stat
        logging.info(f"Remaining URLs in frontier (not crawled): {len(self.frontier)}")

        # Explicitly save frontier and visited_urls_in_queue on normal completion
        self._save(self.path_to_frontier, self.frontier)
        self._save(self.path_to_visited_urls_in_queue, self.visited_urls_in_queue)
        self._save(self.path_to_crawled_data, self.crawled_data)
        self._save(self.path_to_simhashes, self.seen_simhashes)
        logging.info("Frontier, visited URLs, crawled data, and simhashes saved on completion.")


    def _get_robot_parser(self, url):
        parsed_url = urlparse(url)
        domain = parsed_url.netloc

        if domain not in self.robot_parsers:
            robot_url = urlunparse(parsed_url._replace(path="/robots.txt", params="", query="", fragment=""))
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(robot_url)

            current_domain_delay = self.default_delay

            # --- NEW/MODIFIED: Robust fetching for robots.txt ---
            try:
                headers = {'User-Agent': self.user_agent}
                with requests.get(robot_url, headers=headers, timeout=5, stream=True) as response:
                    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                    # Attempt to decode robots.txt content robustly
                    try:
                        robot_content = response.content.decode('utf-8')
                    except UnicodeDecodeError:
                        logging.warning(f"[ROBOTS_ENCODING_ERROR] UnicodeDecodeError for robots.txt of {domain}. Attempting ISO-8859-1 fallback.")
                        robot_content = response.content.decode('iso-8859-1', errors='ignore') # Ignore errors for robots.txt

                rp.parse(robot_content.splitlines())
                logging.info(f"Successfully fetched and parsed robots.txt for {domain}")

                c_delay = rp.crawl_delay(self.user_agent)
                if c_delay is not None:
                    current_domain_delay = c_delay
                    logging.info(f"Applying robots.txt crawl-delay of {c_delay}s for {domain}. Effective delay: {current_domain_delay}s")
                else:
                    logging.info(f"No specific crawl-delay in robots.txt for {domain}. Using default delay: {self.default_delay}s")
                    current_domain_delay = self.default_delay

            except requests.exceptions.HTTPError as e:
                # Handles 404 (robots.txt not found), 403 (forbidden), etc.
                if e.response.status_code in (401, 403):
                    rp.disallow_all = True
                    logging.info(f"Access denied to robots.txt for {domain} (Status: {e.response.status_code}). Disallowing all.")
                elif 400 <= e.response.status_code < 500: # Includes 404
                    rp.allow_all = True
                    logging.info(f"robots.txt not found for {domain} (Status: {e.response.status_code}). Allowing all.")
                else: # Other HTTP errors (e.g., 5xx server errors)
                    rp.allow_all = True
                    logging.error(f"HTTP error fetching robots.txt for {domain} (Status: {e.response.status_code}): {e}. Allowing all.")
            except requests.exceptions.RequestException as e:
                # This catches ConnectionError, Timeout, and any other general request-related issues
                # including potentially a UnicodeDecodeError that might manifest differently.
                logging.warning(f"Failed to fetch robots.txt for {domain} due to: {e}. Allowing all for this domain.")
                rp.allow_all = True # Default to allowing all if there's any issue fetching/parsing robots.txt
            except Exception as e:
                # Catch any other unexpected errors during robots.txt processing (e.g., parsing issues if not from requests)
                logging.error(f"An unexpected error occurred while processing robots.txt for {domain}: {e}. Allowing all for this domain.")
                rp.allow_all = True

            self.robot_parsers[domain] = rp
            self.domain_delays[domain] = current_domain_delay
            # A small delay even after fetching robots.txt to avoid hammering
            time.sleep(self.default_delay)

        return self.robot_parsers[domain]

    def _calculate_priority(self, url, anchor_text, source_page_tokens, current_depth):
            score = 0
            
            # Define base weights for keyword tiers (these are now very significant)
            VERY_RELEVANT_WEIGHT = 70  # Higher impact
            RELEVANT_WEIGHT = 30     # Higher impact
            MODERATELY_RELEVANT_WEIGHT = 15 # Higher impact

            # Domain-based score adjustments
            HIGH_PRIO_DOMAIN_BASE_SCORE = 300 # Significant boost for high priority domains
            BLACKLIST_DOMAIN_PENALTY = -10000 # Effectively puts them at the very end

            # Depth penalty parameters
            BASE_DEPTH_PENALTY_PER_LEVEL = 10
            # For normal/high-prio domains, penalty is linear.
            # For blacklisted domains, they should already be filtered/at end.

            parsed_url = urlparse(url)
            domain = parsed_url.netloc

            # 0. Immediate Filtering for Blacklisted Endings
            if self._is_blacklisted_ending(url):
                return -float('inf') # Guaranteed lowest priority, effectively removed

            # 1. Domain-Tier Prioritization
            if domain in self.blacklisted_domains:
                # If a domain is blacklisted, its priority is extremely low.
                # We still let it through this function to return -float('inf'),
                # but the 'run' loop should ideally filter these early.
                return -float('inf') 
            elif domain in self.high_prio_domains:
                score += HIGH_PRIO_DOMAIN_BASE_SCORE
                # For high-prio domains, maybe a slightly gentler depth penalty,
                # but let the keyword density still drive it if they go off-topic.
                # No specific multiplier here, as keywords should handle it.
            # else: It's a "normal" domain. Its score will be purely built from keyword matches.

            # 2. Anchor Text Relevance (Crucial for "normal" domains)
            if anchor_text:
                cleaned_anchor_text_tokens = set(self._clean_for_scoring(anchor_text))
                
                score += len(cleaned_anchor_text_tokens.intersection(self.very_relevant_keywords)) * VERY_RELEVANT_WEIGHT
                score += len(cleaned_anchor_text_tokens.intersection(self.relevant_keywords)) * RELEVANT_WEIGHT
                score += len(cleaned_anchor_text_tokens.intersection(self.moderately_relevant_keywords)) * MODERATELY_RELEVANT_WEIGHT

                # Penalize for blacklisted city names in anchor text
                for city in self.blacklisted_city_keywords:
                    if city in anchor_text.lower():
                        score -= 50 # Penalize for other major cities
                        break

            # 3. URL Keyword Relevance (path, domain, query) - Also crucial for "normal" domains
            url_components_text = (parsed_url.netloc + parsed_url.path + parsed_url.query).lower()
            url_components_tokens = set(self._clean_for_scoring(url_components_text))
            
            score += len(url_components_tokens.intersection(self.very_relevant_keywords)) * VERY_RELEVANT_WEIGHT
            score += len(url_components_tokens.intersection(self.relevant_keywords)) * RELEVANT_WEIGHT
            score += len(url_components_tokens.intersection(self.moderately_relevant_keywords)) * MODERATELY_RELEVANT_WEIGHT

            # Penalize for blacklisted city names in URL
            for city in self.blacklisted_city_keywords:
                if city in url_components_text:
                    score -= 75
                    break

            # 4. Source Page Context Relevance (still contributes, but less than direct link signals)
            if source_page_tokens:
                source_tokens_set = set(source_page_tokens)
                
                score += len(source_tokens_set.intersection(self.very_relevant_keywords)) * (VERY_RELEVANT_WEIGHT / 3) # Reduced influence
                score += len(source_tokens_set.intersection(self.relevant_keywords)) * (RELEVANT_WEIGHT / 3)
                score += len(source_tokens_set.intersection(self.moderately_relevant_keywords)) * (MODERATELY_RELEVANT_WEIGHT / 3)

                source_page_full_text = " ".join(token for token in source_page_tokens if isinstance(token, str)).lower()
                for city in self.blacklisted_city_keywords:
                    if city in source_page_full_text:
                        score -= 25
                        break
            # 5. Depth Penalty (Linear, but combined with the stronger keyword weights)
            # This will ensure that if a "normal" domain has great keywords at depth 1, it gets high prio,
            # but if it keeps linking to irrelevant stuff, its score quickly drops due to depth AND lack of keywords.
            score -= (current_depth * BASE_DEPTH_PENALTY_PER_LEVEL)

            # Ensure score is not negative if it shouldn't be, before negating for heapq
            # If score is very low but not -inf, it will still be last in the heap.
            return -score # heapq is a min-heap, so we negate for max-priority
        
    def _add_initial_seeds(self, seeds):

        for url in seeds:
            normalized_url, _ = urldefrag(urljoin(url, url))

            # Only add if not already processed in previous runs and not currently in the queue
            if normalized_url not in self.url_to_doc_id and normalized_url not in self.visited_urls_in_queue:
                # Assign an extremely high priority to initial seeds at depth 0
                initial_seed_priority_score = 10000

                # Check if this URL's domain is blacklisted, even for a seed, to avoid wasting time
                # --- NEW/MODIFIED: Robust URL parsing for initial seeds ---
                try:
                    parsed_url = urlparse(normalized_url)
                except Exception as e:
                    logging.warning(f"[SEED_SKIP] Error parsing initial seed URL '{normalized_url}': {e}. Skipping.")
                    continue

                if parsed_url.netloc in self.blacklisted_domains or self._is_blacklisted_ending(normalized_url):
                    logging.warning(f"[SEED_SKIP] Seed '{normalized_url}' domain/ending blacklisted. Skipping initial add.")
                    continue

                # Check robots.txt for the seed URL before adding
                rp = self._get_robot_parser(normalized_url)
                if rp.can_fetch(self.user_agent, normalized_url):
                    # Add to frontier with very high priority
                    heapq.heappush(self.frontier, (-initial_seed_priority_score, normalized_url, 0))
                    self.visited_urls_in_queue.add(normalized_url)
                    logging.info(f"[SEED_ADD] Added new seed to frontier: '{normalized_url}' (Priority: {initial_seed_priority_score}).")
                else:
                    logging.info(f"[SEED_SKIP] Seed '{normalized_url}' disallowed by robots.txt.")

    def _clean_for_scoring(self, text):
        """
        Basic cleaning: lowercase, remove punctuation, normalize whitespace, and return tokens.
        This is used for anchor text, URL components, and source page text for relevance scoring.
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text) # Remove anything that's not alphanumeric or whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text.split() # Return as a list of tokens


    def _get_cleaned_text_for_simhash(self, soup):
        """
        Extracts and cleans text specifically for SimHash calculation.
        This typically involves stripping more boilerplate elements.
        """
        # Exclude common boilerplate tags for better content focus in SimHash
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        text = soup.get_text(separator=" ")

        # Lowercase and remove excessive whitespace for SimHash
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text


    def _compute_simhash(self, text, num_bits=64):
        """
        Computes the SimHash of the given text.
        Uses a simple bag-of-words approach with MD5 hashing for features.
        """
        if not text:
            return 0

        features = {}
        for word in text.split():
            feature_hash = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
            features[feature_hash] = features.get(feature_hash, 0) + 1

        v = [0] * num_bits
        for feature_hash, weight in features.items():
            for i in range(num_bits):
                if (feature_hash >> i) & 1:
                    v[i] += weight
                else:
                    v[i] -= weight

        simhash_val = 0
        for i in range(num_bits):
            if v[i] >= 0:
                simhash_val |= (1 << i)
        return simhash_val


    def _hamming_distance(self, hash1, hash2):
        """Calculates the Hamming distance between two integers (representing hashes)."""
        x = hash1 ^ hash2
        set_bits = 0
        while x > 0:
            x &= (x - 1)
            set_bits += 1
        return set_bits


    def _get_id(self, url):
        # Use existing ID if URL was previously crawled and saved
        if url in self.url_to_doc_id:
            return self.url_to_doc_id[url]

        # Otherwise, assign a new ID
        if self.crawled_data:
            return max(self.crawled_data.keys()) + 1
        return 0 # First document will be ID 0

    # TODO these three methods should be unified
    def _save_crawled(self, doc_id, doc_info):
        self.crawled_data[doc_id] = doc_info

        with open(self.path_to_crawled_data, "wb") as f:
            pickle.dump(self.crawled_data, f)


    def _save(self, path, data):
        """Generic save method for any pickleable data."""
        try:
            with open(path, "wb") as f:
                pickle.dump(data, f)
            logging.info(f"Successfully saved data to {path}.")
        except Exception as e:
            # NEW/MODIFIED: Catch any error during saving
            logging.error(f"Error saving data to {path}: {e}. Data might be lost!")


    def _save_all(self):
        with open(self.path_to_crawled_data, "wb") as f:
            pickle.dump(self.crawled_data, f)

    def _load(self, path):
        """Generic load method for any pickleable data, with robust error handling."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            logging.info(f"Loaded existing data from {path}")
            return data
        except FileNotFoundError:
            logging.warning(f"No previous data found for {path}. Initializing empty.")
            # NEW/MODIFIED: Ensure correct empty data structure is returned for each path
            if path == self.path_to_crawled_data:
                return {}
            elif path == self.path_to_simhashes:
                return set()
            elif path == self.path_to_frontier:
                return []
            elif path == self.path_to_visited_urls_in_queue:
                return set()
            else:
                logging.error(f"Attempted to load unrecognized path: {path}")
                raise # Re-raise if path is not handled, indicating a logic error
        except (pickle.UnpicklingError, EOFError) as e: # NEW/MODIFIED: Catch EOFError for truncated files
            logging.error(f"Error unpickling data from {path}: {e}. Initializing empty data structures to prevent further errors.")
            # Return appropriate empty structures to allow the crawler to continue
            if path == self.path_to_crawled_data:
                return {}
            elif path == self.path_to_simhashes:
                return set()
            elif path == self.path_to_frontier:
                return []
            elif path == self.path_to_visited_urls_in_queue:
                return set()
            else:
                raise # Re-raise if path is not handled
        except Exception as e: # NEW/MODIFIED: Catch any other unexpected errors during loading
            logging.error(f"An unexpected error occurred while loading data from {path}: {e}. Initializing empty data structures.")
            # Return appropriate empty structures
            if path == self.path_to_crawled_data:
                return {}
            elif path == self.path_to_simhashes:
                return set()
            elif path == self.path_to_frontier:
                return []
            elif path == self.path_to_visited_urls_in_queue:
                return set()
            else:
                raise # Re-raise if path is not handled

    def _handle_interrupt(self, signum, frame):
        logging.info("\n[INTERRUPT] Ctrl+C detected. Saving state and shutting down gracefully...")
        try:
            # Use the robust _save method for all state saving
            self._save(self.path_to_frontier, self.frontier)
            self._save(self.path_to_visited_urls_in_queue, self.visited_urls_in_queue)
            self._save(self.path_to_crawled_data, self.crawled_data)
            self._save(self.path_to_simhashes, self.seen_simhashes)
            logging.info("Crawler state saved successfully.")
        except Exception as e:
            logging.error(f"Error during graceful shutdown save: {e}. Data might be partially lost!")
        sys.exit(0)

    def _is_blacklisted_ending(self, url): # <--- THIS METHOD NEEDS TO BE INDENTED
        """Checks if the URL ends with any blacklisted extension or string."""
        url_lower = url.lower()
        for ending in self.url_ending_blacklist:
            if url_lower.endswith(ending):
                return True
        return False

# if __name__ == "__main__":
    
#     seeds = [
#     # Official & Governmental
#     "https://www.tuebingen.de/en/",
#     "https://www.tuebingen.de/en/wirtschaft.html",
#     "https://www.tuebingen-info.de/",
#     "https://www.germany.travel/en/cities-culture/tuebingen.html",
#     "https://historicgermany.travel/historic-germany/tubingen/",
#     "https://www.visit-bw.com/en/article/tubingen/df9223e2-70e5-4ee9-b3f2-cd2355ab8551",
#     "https://www.tourism-bw.com/city/tuebingen-1b7f535ace",

#     # Academic & Research
#     "https://uni-tuebingen.de/en/",
#     "https://www.welcome.uni-tuebingen.de/",
#     "https://tuebingenresearchcampus.com/en/",
#     "https://www.mastersportal.com/universities/188/university-of-tbingen.html",
#     "https://www.expatrio.com/about-germany/eberhard-karls-universitat-tubingen",
#     "https://www.uni-tuebingen.de/en/faculties/",
#     "https://www.zmbp.uni-tuebingen.de/en/",
#     "https://www.nmi.de/en/",
#     "https://www.tuebingen.mpg.de/en/",
#     "https://www.hertie-institute.com/en/home/",
#     "https://www.cil-tuebingen.de/en/home/",
#     "https://cyber-valley.de/en/",
#     "https://tuebingen.ai/",
#     "https://www.daad.de/en/studying-in-germany/universities/university-profiles/uni-detail/1134/",
#     "https://www.bachelorsportal.com/universities/188/university-of-tbingen.html",
#     "https://students.tufts.edu/tufts-global-education/explore/semesteryear-programs/tufts-programs/tufts-tubingen",

#     # Tourism & Travel Guides
#     "https://en.wikipedia.org/wiki/T%C3%BCbingen",
#     "https://en.wikivoyage.org/wiki/T%C3%BCbingen",
#     "https://www.tripadvisor.com/Attractions-g198539-Activities-Tubingen_Baden_Wurttemberg.html",
#     "https://www.tripadvisor.com/Tourism-g198539-Tubingen_Baden_Wurttemberg-Vacations.html",
#     "https://www.europeanbestdestinations.com/destinations/tubingen/",
#     "https://theculturetrip.com/europe/germany/articles/the-best-things-to-see-and-do-in-tubingen-germany",
#     "https://www.mygermanyvacation.com/best-things-to-do-and-see-in-tubingen-germany/",
#     "https://justinpluslauren.com/things-to-do-in-tubingen-germany/",
#     "https://simplifylivelove.com/tubingen-germany/",
#     "https://www.expedia.com/Things-To-Do-In-Tubingen.d55289.Travel-Guide-Activities",
#     "https://www.lonelyplanet.com/germany/baden-wurttemberg/tubingen",
#     "https://www.try-travel.com/blog/europe/germany/tubingen/things-to-do-in-tubingen/",
#     "https://velvetescape.com/things-to-do-in-tubingen/",
#     "https://thedesigntourist.com/12-top-things-to-do-in-tubingen-germany/",
#     "https://globaltravelescapades.com/things-to-do-in-tubingen-germany/",
#     "https://thespicyjourney.com/magical-things-to-do-in-tubingen-in-one-day-tuebingen-germany-travel-guide/",
#     "https://veganfamilyadventures.com/15-best-things-to-do-in-tubingen-germany/",
#     "https://thetouristchecklist.com/things-to-do-in-tubingen/",
#     "https://visit-tubingen.co.uk/",
#     "https://visit-tubingen.co.uk/welcome-to-tubingen/",
#     "https://www.germansights.com/tubingen/",
#     "https://www.atlasobscura.com/things-to-do/tubingen-germany",
#     "https://angiestravelroutes.com/en/12of12-tuebingen-sightseeing/",
#     "https://explorial.com/uncover-tubingens-hidden-gems-10-lesser-known-spots/",

#     # Attractions & Things to Do
#     "https://things.in/germany/t%C3%BCbingen",
#     "https://www.komoot.com/guide/210692/attractions-around-landkreis-tuebingen",
#     "https://touristplaces.guide/top-10-places-to-visit-in-tubingen-nature-adventure-and-history/",
#     "https://www.getyourguide.com/tubingen-l150729/actividades-tc54/",
#     "https://www.viamichelin.com/maps/tourist-attractions/germany/baden_wurttemberg/tubingen/tubingen-72070",
#     "https://www.tripadvisor.com/Attractions-g198539-Activities-zft11306-Tubingen_Baden_Wurttemberg.html",
#     "https://tropter.com/en/germany/tubingen",
#     "https://www.ostrichtrails.com/europe/germany/tubingen-walking-tour/",
#     "https://www.keeptravel.com/germany/tuebingen",
#     "https://www.trip.com/travel-guide/attraction/tubingen-44519/tourist-attractions/",
#     "https://www.neuralword.com/en/travel-tourism/europe/exploring-tubingen-must-see-attractions-and-sites",
#     "https://www.fodors.com/world/europe/germany/heidelberg-and-the-neckar-valley/places/tubingen/things-to-do/sights",
#     "https://www.visitacity.com/en/tubingen/attraction-by-type/all-attractions",
#     "https://www.visit-bw.com/en/article/old-town-of-tubingen/c82ceb67-f78d-4b4e-8911-3972ca794cbc#/",
#     "https://www.triphobo.com/places/t-bingen-germany/things-to-do",
#     "https://triplyzer.com/germany/things-to-do-in-tubingen/",
#     "https://things.in/germany/t%C3%BCbingen",
#     "https://www.komoot.com/guide/210692/attractions-around-landkreis-tuebingen",
#     "https://touristplaces.guide/top-10-places-to-visit-in-tubingen-nature-adventure-and-history/",
#     "https://www.getyourguide.com/tubingen-l150729/actividades-tc54/",
#     "https://www.tuebingen-info.de/de/tuebinger-flair/sehenswuerdigkeiten",
#     "https://www.viamichelin.com/maps/tourist-attractions/germany/baden_wurttemberg/tubingen/tubingen-72070",
#     "https://wanderboat.ai/localities/tuebingen/kzRxWg6tQGiZh5DLUhdxNg",
#     "https://www.tripadvisor.com/Attractions-g198539-Activities-zft11306-Tubingen_Baden_Wurttemberg.html",
#     "https://www.tripadvisor.ie/Attractions-g198539-Activities-Tubingen_Baden_Wurttemberg.html",
#     "https://en.wikivoyage.org/wiki/T%C3%BCbingen",
#     "https://tropter.com/en/germany/tubingen",
#     "https://www.ostrichtrails.com/europe/germany/tubingen-walking-tour/",
#     "https://www.keeptravel.com/germany/tuebingen",
#     "https://www.trip.com/travel-guide/attraction/tubingen-44519/tourist-attractions/",
#     "https://www.komoot.com/guide/210692/de-mooiste-attracties-rond-tuebingen",
#     "https://www.alltrails.com/trail/germany/baden-wurttemberg/von-boblingen-nach-tubingen",
#     "https://www.neuralword.com/en/travel-tourism/europe/exploring-tubingen-must-see-attractions-and-sites",
#     "https://pineqone.com/attractions/stadtmuseum-tubingen/",
#     "https://www.atlasobscura.com/things-to-do/tubingen-germany",
#     "https://www.try-travel.com/category/europe/germany/tubingen/",
#     "https://attractions.pw/blog/tubingen/",
#     "https://evendo.com/locations/germany/hohenzollern-castle/attraction/st-george-s-collegiate-church-tubingen",
#     "https://tripomatic.com/en/list/top-tourist-attractions-in-regierungsbezirk-tubingen-region:13332",
#     "www.thecrazytourist.com/15-best-things-reutlingen-germany/",
#     "https://freiburger-bote.de/freizeit/freizeitaktivitaeten-tuebingen/",
#     "https://www.krone-tuebingen.de/en/a-holiday-in-tuebingen/tuebingen-the-surrounding-area/",
#     "https://www.germansights.com/tubingen/",
#     "https://www.fodors.com/world/europe/germany/heidelberg-and-the-neckar-valley/places/tubingen/things-to-do/sights",
#     "https://www.visitacity.com/en/tubingen/attraction-by-type/all-attractions",
#     "https://angiestravelroutes.com/en/12of12-tuebingen-sightseeing/",
#     "https://uni-tuebingen.de/en/faculties/faculty-of-economics-and-social-sciences/subjects/department-of-social-sciences/methods-center/events/past-events/fgme2017/tuebingen-1/attractions/",
#     "https://visit-tubingen.co.uk/category/things-to-do-in-tubingen/",
#     "https://www.outdooractive.com/en/travel-guide/germany/tuebingen/1022199/",
#     "https://www.visit-bw.com/en/article/old-town-of-tubingen/c82ceb67-f78d-4b4e-8911-3972ca794cbc#/",
#     "https://www.triphobo.com/places/t-bingen-germany/things-to-do",
#     "https://uni-tuebingen.de/forschung/zentren-und-institute/brasilien-und-lateinamerika-zentrum/german-brazilian-symposium-2024/about-tuebingen/welcome-to-tuebingen/",
#     "https://www.expedia.com/Things-To-Do-In-Tuebingen.d181220.Travel-Guide-Activities",
#     "https://triplyzer.com/germany/things-to-do-in-tubingen/",
#     "https://explorial.com/uncover-tubingens-hidden-gems-10-lesser-known-spots/",
#     "https://www.tripadvisor.de/Restaurants-g198539-Tubingen_Baden_Wurttemberg.html",
#     "https://www.tuebingen-info.de/de/mein-aufenthalt/gastronomie/restaurants-und-gaststaetten",
#     "https://www.reservix.de/sport-in-tuebingen?_locale=en",
#     "https://www.outdooractive.com/en/sports/tuebingen/athletes-destinations-in-tuebingen/21876962/",
#     "https://uni-tuebingen.de/en/facilities/central-institutions/university-sports-center/sports-program/",
#     "https://uni-tuebingen.de/en/faculties/faculty-of-economics-and-social-sciences/subjects/department-of-social-sciences/sports-science/institute/",
#     "https://www.iwm-tuebingen.de/en/research/projects/Vertrauen?name=Vertrauen",

#     # Gastronomy & Restaurants
#     "https://www.tripadvisor.com/Restaurants-g198539-Tubingen_Baden_Wurttemberg.html",
#     "https://guide.michelin.com/en/baden-wurttemberg/tubingen/restaurants",
#     "https://www.outdooractive.com/en/eat-and-drink/tuebingen/eat-and-drink-in-tuebingen/21873363/",
#     "https://www.opentable.com/food-near-me/stadt-tubingen-germany",
#     "https://www.yelp.com/search?find_desc=Restaurants&find_loc=T%C3%BCbingen",
#     "https://www.happycow.net/europe/germany/tubingen/",
#     "https://www.wurstkueche.com/en/restaurant-our-place/",
#     "https://www.1821tuebingen.de/",
#     "https://www.lacasa-tuebingen.de/en/index.php",
#     "https://www.restaurant-waldhorn.de/en/",
#     "https://www.maugan.de/en/",
#     "https://en.wikivoyage.org/wiki/T%C3%BCbingen#Eat",
#     "https://en.wikivoyage.org/wiki/T%C3%BCbingen#Drink",
#     "https://listium.com/@thedesigntourist/98696/15-must-visit-spots-in-tbingen-germany",
#     "https://restaurantguru.com/Tubingen#restaurant-list",
#     "https://www.kikijourney.com/tuebingen-christmas-market-germany-the-ultimate-guide/",
#     "https://www.getyourguide.com/tubingen-l150729/genussvoll-durch-tubingen-kulinarische-stadtfuhrung-t644161/",
#     "https://www.faros-tuebingen.com/",
#     "https://christmas-events-near-me.com/tuebingen-germany/",
#     "https://www.my-stuwe.de/en/refectory/",
#     "https://www.eventbrite.com/b/germany--t%C3%BCbingen/food-and-drink/",
#     "https://wanderlog.com/list/geoCategory/199488/where-to-eat-best-restaurants-in-tubingen",
#     "https://wanderlog.com/list/geoCategory/312176/best-spots-for-lunch-in-tubingen",
#     "https://www.tripadvisor.com/Restaurants-g198539-Tubingen_Baden_Wurttemberg.html",
#     "https://www.tripadvisor.com/Restaurants-g198539-zfp58-Tubingen_Baden_Wurttemberg.html",
#     "https://guide.michelin.com/en/baden-wurttemberg/tubingen/restaurants",
#     "https://guide.michelin.com/us/en/baden-wurttemberg/tbingen/restaurants",
#     "https://www.outdooractive.com/en/eat-and-drink/tuebingen/eat-and-drink-in-tuebingen/21873363/",
#     "https://www.opentable.com/food-near-me/stadt-tubingen-germany",
#     "https://www.yelp.com/search?find_desc=Restaurants&find_loc=T%C3%BCbingen",
#     "https://m.yelp.com/search?cflt=swabian&find_loc=T%C3%BCbingen%2C+Baden-W%C3%BCrttemberg",
#     "https://www.happycow.net/europe/germany/tubingen/",
#     "https://www.wurstkueche.com/en/restaurant-our-place/",
#     "https://www.1821tuebingen.de/",
#     "https://www.lacasa-tuebingen.de/en/index.php",
#     "https://www.restaurant-waldhorn.de/en/",
#     "https://www.maugan.de/en/",
#     "https://en.wikivoyage.org/wiki/T%C3%BCbingen#Eat",
#     "https://en.wikivoyage.org/wiki/T%C3%BCbingen#Drink",
#     "https://listium.com/@thedesigntourist/98696/15-must-visit-spots-in-tbingen-germany",
#     "https://www.tripadvisor.com/Restaurants-g198539-Tubingen_Baden_Wurttemberg.html",
#     "https://restaurantguru.com/Tubingen#restaurant-list",
#     "https://www.kikijourney.com/tuebingen-christmas-market-germany-the-ultimate-guide/",
#     "https://www.getyourguide.com/tubingen-l150729/genussvoll-durch-tubingen-kulinarische-stadtfuhrung-t644161/",
#     "https://www.faros-tuebingen.com/",
#     "https://christmas-events-near-me.com/tuebingen-germany/",
#     "https://uni-tuebingen.de/en/international/welcome-center/registration/",
#     "https://www.my-stuwe.de/en/refectory/",
#     "https://www.opentable.com/food-near-me/stadt-tubingen-germany",
#     "https://www.eventbrite.com/b/germany--t%C3%BCbingen/food-and-drink/",
#     "https://www.yelp.com/search?find_desc=Drink&find_loc=T%C3%BCbingen%2C+Baden-W%C3%BCrttemberg",
#     "https://www.ubereats.com/de-en/city/t%C3%BCbingen-bw?srsltid=AfmBOooVHU1cR4Js1VRCXuua2RVKE36M4jfq-FmEbFDbrMJojWYBNwSY",
#     "https://www.my-stuwe.de/en/refectory/cafeteria-unibibliothek-tuebingen/",
#     "https://www.tripadvisor.com/Restaurant_Review-g198539-d3193708-Reviews-Kuckuck-Tubingen_Baden_Wurttemberg.html",
#     "https://www.wurstkueche.com/en/restaurant-our-place/",
#     "https://www.mph.tuebingen.mpg.de/canteen",
#     "https://www.reddit.com/r/Tuebingen/comments/10s7frj/best_burger_place_in_tubingen/",
#     "https://neckawa.de/wp-content/uploads/2024/07/2024_07_05-Speisekarten_Neckawa-web.pdf",
#     "https://uni-tuebingen.de/en/international/sprachen-lernen/deutschkurse/suedafrika-programm/programm-2017/tuebingen-diary/german-food/",
#     "https://www.krone-tuebingen.de/en/restaurants/restaurant-ludwigs/",
#     "https://www.outdooractive.com/en/places-to-eat-and-drink/tuebingen/eat-and-drink-in-tuebingen/21873363/",
#     "https://uni-tuebingen.de/fakultaeten/philosophische-fakultaet/fachbereiche/geschichtswissenschaft/seminareinstitute/neuere-geschichte/studium/lehrveranstaltungen/civis-classes/",
#     "https://www.lieferando.de/en/delivery/food/tuebingen-72072"

#     # Local News & Community (English)
#     "https://integreat.app/tuebingen/en/news/tu-news",
#     "https://tunewsinternational.com/category/news-in-english/",
#     "https://www.reddit.com/r/Tuebingen/",
#     "https://www.reddit.com/r/Tuebingen/comments/1rhpyk/life_as_an_english_speaking_person_in_t%C3%BCbingen/",
#     "https://www.meetup.com/tubingen-meet-mingle/",
#     "https://www.internations.org/germany-expats/",
#     "https://www.facebook.com/groups/tuebingenexpats/",
#     "https://www.couchsurfing.com/places/europe/germany/tubingen",

#     # Accommodation
#     "https://www.booking.com/city/de/tubingen.html",
#     "https://all.accor.com/a/en/destination/city/hotels-tubingen-v4084.html",
#     "https://www.hotel-am-schloss.de/en/",
#     "https://www.ibis.com/gb/hotel-3200-ibis-tuebingen/index.shtml",
#     "https://www.kronprinz-tuebingen.de/en/",

#     # Culture & Arts / Events
#     "https://www.stadtmuseum-tuebingen.de/english/",
#     "https://www.kunsthallentuebingen.de/en/",
#     "https://www.landestheater-tuebingen.de/en/home/",
#     "https://www.franzk.net/en/",
#     "https://www.kino-arsenal.de/",
#     "https://www.tuebinger-kultursommer.de/",
#     "https://www.tuebinger-stadtlauf.de/en/home/",
#     "https://rausgegangen.de/en/tubingen/",
#     "https://www.bandsintown.com/c/tuebingen-germany",
#     "https://www.eventbrite.com/d/germany--t%C3%BCbingen/english/",

#     # Outdoor & Recreation
#     "https://www.alltrails.com/germany/baden-wurttemberg/tubingen",
#     "https://www.outdooractive.com/en/city-walks/tuebingen/city-walks-in-tuebingen/8232815/",
#     "https://www.outdooractive.com/en/hiking-trails/tuebingen/hiking-in-tuebingen/1432855/",
#     "https://www.komoot.com/guide/881/hiking-around-landkreis-tuebingen",
#     "https://www.wikiloc.com/trails/hiking/germany/baden-wurttemberg/tubingen",
#     "https://www.outdooractive.com/en/travel-guide/germany/tuebingen/1022199/",

#     # Sports & Recreation
#     "https://www.reservix.de/sport-in-tuebingen?_locale=en",
#     "https://www.outdooractive.com/en/sports/tuebingen/athletes-destinations-in-tuebingen/21876962/",
#     "https://uni-tuebingen.de/en/facilities/central-institutions/university-sports-center/",
#     "https://uni-tuebingen.de/en/facilities/central-institutions/university-sports-center/sports-program/",
#     "https://www.sportzentrum.uni-tuebingen.de/en/",
#     "https://www.outdooractive.com/en/swimming-pools/tuebingen/swimming-in-tuebingen/21875473/",
#     "https://www.outdooractive.com/en/fitness-centres/tuebingen/fitness-centres-in-tuebingen/21875484/",

#     # Business & Local Economy
#     "https://www.hk-reutlingen.de/en/start/",

#     # Shopping & Commerce
#     "https://tuebingenresearchcampus.com/en/tuebingen/living-in-tuebingen/shopping",
#     "https://www.tripadvisor.com/Attractions-g198539-Activities-c26-Tubingen_Baden_Wurttemberg.html",
#     "https://cityseeker.com/tuebingen-de/shopping",
#     "https://www.outdooractive.com/en/shoppings/tuebingen/shopping-in-tuebingen/21876964/",
#     "https://www.yelp.com/search?cflt=shopping&find_loc=T%C3%BCbingen%2C+Baden-W%C3%BCrttemberg",
#     "https://uni-tuebingen.de/international/studierende-aus-dem-ausland/erasmus-und-austausch-nach-tuebingen/studentisches-leben/tuebingen-basics-and-beyond/living-in-tuebingen/",
#     "https://uni-tuebingen.de/en/280707",
#     "https://us.trip.com/travel-guide/shops/city-44519/",

#     # Transportation & Mobility
#     "https://tuebingenresearchcampus.com/en/tuebingen/living-in-tuebingen/mobility/by-public-transport",
#     "https://www.bahnhof.de/en/tuebingen-hbf/parking-spaces",
#     "https://www.swtue.de/en/private-customer/parking.html",
#     "https://www.tripadvisor.com/ShowTopic-g198539-i4335-k5492275-How_can_I_use_the_bus_lines-Tubingen_Baden_Wurttemberg.html",
#     "https://tuebingen.ai/wiki/transportation-and-mobility",
#     "https://www.bahnhof.de/en/tuebingen-hbf/map",
#     "https://uni-tuebingen.de/en/university/how-to-get-here/",
#     "https://en.parkopedia.com/parking/t%C3%BCbingen/",
#     "https://www.tuebingen-info.de/en/service/vor-ort/mit-dem-auto",

#     # Health & Medical Services
#     "https://www.uniklinik-tuebingen.de/en/",
#     "https://tuebingenresearchcampus.com/en/tuebingen/living-in-tuebingen/healthcare",
#     "https://uni-tuebingen.de/en/280706",
#     "https://www.doctolib.de/city/tuebingen",
#     "https://www.kliniksuche.de/city/tuebingen",

#     # Real Estate & Housing
#     "https://www.studenten-wg.de/city/tuebingen",
#     "https://www.wg-gesucht.de/en/wg-zimmer-in-Tuebingen.125.0.1.0.html",
#     "https://www.immobilienscout24.de/Suche/de/baden-wuerttemberg/tuebingen",
#     "https://tuebingenresearchcampus.com/en/tuebingen/living-in-tuebingen/housing",
#     "https://www.welcome.uni-tuebingen.de/housing/",
#     "https://uni-tuebingen.de/en/study/student-life/student-housing/",
#     "https://uni-tuebingen.de/en/international/study-in-tuebingen/degree-seeking-students/getting-started-and-orientation-for-international-students/housing/",
#     "https://www.my-stuwe.de/en/housing/halls-of-residence-tuebingen/",
#     "https://tuebingenresearchcampus.com/en/tuebingen/accommodations/finding-a-home/housing-options",

#     # Education (Non-University)
#     "https://www.goethe.de/ins/de/en/sta/tue.html",
#     "https://www.berlitz.com/locations/germany/tuebingen",
#     "https://tuebingenresearchcampus.com/en/tuebingen/living-in-tuebingen/language-courses",

#     # Services & Utilities
#     "https://tuebingenresearchcampus.com/en/tuebingen/living-in-tuebingen/banking",
#     "https://tuebingenresearchcampus.com/en/tuebingen/living-in-tuebingen/internet-and-mobile",
#     "https://www.swtue.de/en/",
#     "https://www.deutsche-post.de/en/branch-finder.html",

#     # Student Life & University Services
#     "https://uni-tuebingen.de/en/study/student-life/",
#     "https://www.my-stuwe.de/en/refectory/cafeteria-unibibliothek-tuebingen/",
#     "https://www.mph.tuebingen.mpg.de/canteen",
#     "https://uni-tuebingen.de/en/international/welcome-center/registration/",

#     # Forums
#     "https://uni-tuebingen.de/forschung/zentren-und-institute/tuebinger-forum-fuer-wissenschaftskulturen-tfw/tfw/",
#     "https://uni-tuebingen.de/forschung/zentren-und-institute/forum-scientiarum/",
#     "https://www.studis-online.de/Fragen-Brett/list.php?84",
#     "https://stadtistik.de/stadt/tuebingen-08416041/",
#     "https://www.juristischesforum.de/",
#     "https://www.studis-online.de/Fragen-Brett/read.php?84,701326,1969302,quote=1",
#     "https://www.tuebingen.de/Dateien/Praesentation5Altstadtforum.pdf",
#     "https://uni-tuebingen.de/en/excellence-strategy/research/platforms/global-encounters/tuebingen-forum-on-social-resonances-of-societal-crises-tueforce/",
#     "https://erasmusu.com/en/erasmus-tubingen/erasmus-forum",
#     "https://www.rosalux.de/en/publication/id/2539/1-volxuni-des-social-forum-tuebingen-reutlingen",
#     "https://www.kulturforum.info/de/forum-partner/alle-partner/6016-1001121-institut-fuer-osteuropaeische-geschichte-und-landeskunde-an-der-eberhard-karls-universitaet-tuebingen",
#     "https://www.tripadvisor.com/ShowForum-g198539-i4335-Tubingen_Baden_Wurttemberg.html",
#     "https://uni-tuebingen.de/en/research/centers-and-institutes/tuebingen-forum-for-science-and-humanities/tfw/",
#     "https://www.reddit.com/r/Tuebingen/",
#     "https://ps-forum.cs.uni-tuebingen.de/login",
#     "https://www.reddit.com/r/UniTuebingen/",
#     "https://uni-tuebingen.de/en/research/centers-and-institutes/tuebingen-forum-for-science-and-humanities/about/institution/",
#     "https://www.facebook.com/FreigeistForumTubingen/posts/hier-gesammelte-vortr%C3%A4ge-und-interviewshttpwwwfreigeist-forum-tuebingendepvortra/815537238534598/",
#     "https://community.ricksteves.com/travel-forum/germany-reviews/tubingen",
#     "https://www.n-k-t.de/"
# ]
                    

#     # Initialize the crawler
#     # max_depth: How many layers deep the crawler will go. Start with 1 or 2 for testing.
#     # delay: Delay between requests to the same domain (in seconds). Adjust as needed.
#     # simhash_threshold: How similar content can be before being considered a duplicate.
#     crawler = OfflineCrawler(
#         seeds=seeds,
#         max_depth=4,  # Start with a low depth for quick tests
#         delay=0.5,      # Be polite with the crawl delay
#         simhash_threshold=2
#     )

#     # Run the crawler
#     crawler.run()
    
#     print("\nCrawler finished.")

