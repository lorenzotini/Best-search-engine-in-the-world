from Utils.legal_crawling import OfflineCrawler
from Utils.indexer import Indexer
from Utils.hybrid_retrieval import HybridRetrieval
from Utils.query_expander import QueryExpander
from Utils.text_preprocessor import preprocess_text
import time
from transformers import pipeline

def search(query_text: str, 
           indexer: Indexer, 
           hybrid_model:HybridRetrieval, 
           use_query_expansion=True):
    
    start = time.time()
    query_tokens = preprocess_text(query_text, isQuery=True)
    end = time.time()
    print("Time to preprocess query: ", end - start)
    
    start = time.time()
    if use_query_expansion:
        print("\nExpanding query...\n")
        expander = QueryExpander(max_synonyms=2, synonym_weight=0.25, original_weight=1.0)
        weighted_tokens = expander.expand(query_tokens)
        print("Expanded query tokens with weights:", weighted_tokens)
    else:
        weighted_tokens = [(token, 1.0) for token in query_tokens]

    end = time.time()
    print("Time to expand query: ", end - start)

    original_terms = [term for term, weight in weighted_tokens if weight >= 1.0]
    candidates_ids = indexer.get_union_candidates(original_terms)

    print("candidate size = ", len(candidates_ids))

    if not candidates_ids:
        return []
    
    print("\nUsing hybrid model...\n")
    start = time.time()
    results = hybrid_model.retrieve(weighted_tokens, candidates_ids)
    end = time.time()
    print("Time to rank: ", end - start)

    return results

def initialize_crawling(seeds):
    print("Crawling...")
    crawler = OfflineCrawler(seeds, max_depth=4)
    crawler.run()
    
    #print("Indexing...")
    # indexer = Indexer()
    #indexer.run()

    # print("Initializing models...")
    # hybrid_model = HybridRetrieval()

    seeds = [
    # Official & Governmental
    "https://www.tuebingen.de/en/",
    "https://www.tuebingen.de/en/wirtschaft.html",
    "https://www.tuebingen-info.de/",
    "https://www.germany.travel/en/cities-culture/tuebingen.html",
    "https://historicgermany.travel/historic-germany/tubingen/",
    "https://www.visit-bw.com/en/article/tubingen/df9223e2-70e5-4ee9-b3f2-cd2355ab8551",
    "https://www.tourism-bw.com/city/tuebingen-1b7f535ace",

    # Academic & Research
    "https://uni-tuebingen.de/en/",
    "https://www.welcome.uni-tuebingen.de/",
    "https://tuebingenresearchcampus.com/en/",
    "https://www.mastersportal.com/universities/188/university-of-tbingen.html",
    "https://www.expatrio.com/about-germany/eberhard-karls-universitat-tubingen",
    "https://www.uni-tuebingen.de/en/faculties/",
    "https://www.zmbp.uni-tuebingen.de/en/",
    "https://www.nmi.de/en/",
    "https://www.tuebingen.mpg.de/en/",
    "https://www.hertie-institute.com/en/home/",
    "https://www.cil-tuebingen.de/en/home/",
    "https://cyber-valley.de/en/",
    "https://tuebingen.ai/",
    "https://www.daad.de/en/studying-in-germany/universities/university-profiles/uni-detail/1134/",
    "https://www.bachelorsportal.com/universities/188/university-of-tbingen.html",
    "https://students.tufts.edu/tufts-global-education/explore/semesteryear-programs/tufts-programs/tufts-tubingen",

    # Tourism & Travel Guides
    "https://en.wikipedia.org/wiki/T%C3%BCbingen",
    "https://en.wikivoyage.org/wiki/T%C3%BCbingen",
    "https://www.tripadvisor.com/Attractions-g198539-Activities-Tubingen_Baden_Wurttemberg.html",
    "https://www.tripadvisor.com/Tourism-g198539-Tubingen_Baden_Wurttemberg-Vacations.html",
    "https://www.europeanbestdestinations.com/destinations/tubingen/",
    "https://theculturetrip.com/europe/germany/articles/the-best-things-to-see-and-do-in-tubingen-germany",
    "https://www.mygermanyvacation.com/best-things-to-do-and-see-in-tubingen-germany/",
    "https://justinpluslauren.com/things-to-do-in-tubingen-germany/",
    "https://simplifylivelove.com/tubingen-germany/",
    "https://www.expedia.com/Things-To-Do-In-Tubingen.d55289.Travel-Guide-Activities",
    "https://www.lonelyplanet.com/germany/baden-wurttemberg/tubingen",
    "https://www.try-travel.com/blog/europe/germany/tubingen/things-to-do-in-tubingen/",
    "https://velvetescape.com/things-to-do-in-tubingen/",
    "https://thedesigntourist.com/12-top-things-to-do-in-tubingen-germany/",
    "https://globaltravelescapades.com/things-to-do-in-tubingen-germany/",
    "https://thespicyjourney.com/magical-things-to-do-in-tubingen-in-one-day-tuebingen-germany-travel-guide/",
    "https://veganfamilyadventures.com/15-best-things-to-do-in-tubingen-germany/",
    "https://thetouristchecklist.com/things-to-do-in-tubingen/",
    "https://visit-tubingen.co.uk/",
    "https://visit-tubingen.co.uk/welcome-to-tubingen/",
    "https://www.germansights.com/tubingen/",
    "https://www.atlasobscura.com/things-to-do/tubingen-germany",
    "https://angiestravelroutes.com/en/12of12-tuebingen-sightseeing/",
    "https://explorial.com/uncover-tubingens-hidden-gems-10-lesser-known-spots/",

    # Attractions & Things to Do
    "https://things.in/germany/t%C3%BCbingen",
    "https://www.komoot.com/guide/210692/attractions-around-landkreis-tuebingen",
    "https://touristplaces.guide/top-10-places-to-visit-in-tubingen-nature-adventure-and-history/",
    "https://www.getyourguide.com/tubingen-l150729/actividades-tc54/",
    "https://www.viamichelin.com/maps/tourist-attractions/germany/baden_wurttemberg/tubingen/tubingen-72070",
    "https://www.tripadvisor.com/Attractions-g198539-Activities-zft11306-Tubingen_Baden_Wurttemberg.html",
    "https://tropter.com/en/germany/tubingen",
    "https://www.ostrichtrails.com/europe/germany/tubingen-walking-tour/",
    "https://www.keeptravel.com/germany/tuebingen",
    "https://www.trip.com/travel-guide/attraction/tubingen-44519/tourist-attractions/",
    "https://www.neuralword.com/en/travel-tourism/europe/exploring-tubingen-must-see-attractions-and-sites",
    "https://www.fodors.com/world/europe/germany/heidelberg-and-the-neckar-valley/places/tubingen/things-to-do/sights",
    "https://www.visitacity.com/en/tubingen/attraction-by-type/all-attractions",
    "https://www.visit-bw.com/en/article/old-town-of-tubingen/c82ceb67-f78d-4b4e-8911-3972ca794cbc#/",
    "https://www.triphobo.com/places/t-bingen-germany/things-to-do",
    "https://triplyzer.com/germany/things-to-do-in-tubingen/",
    "https://things.in/germany/t%C3%BCbingen",
    "https://www.komoot.com/guide/210692/attractions-around-landkreis-tuebingen",
    "https://touristplaces.guide/top-10-places-to-visit-in-tubingen-nature-adventure-and-history/",
    "https://www.getyourguide.com/tubingen-l150729/actividades-tc54/",
    "https://www.tuebingen-info.de/de/tuebinger-flair/sehenswuerdigkeiten",
    "https://www.viamichelin.com/maps/tourist-attractions/germany/baden_wurttemberg/tubingen/tubingen-72070",
    "https://wanderboat.ai/localities/tuebingen/kzRxWg6tQGiZh5DLUhdxNg",
    "https://www.tripadvisor.com/Attractions-g198539-Activities-zft11306-Tubingen_Baden_Wurttemberg.html",
    "https://www.tripadvisor.ie/Attractions-g198539-Activities-Tubingen_Baden_Wurttemberg.html",
    "https://en.wikivoyage.org/wiki/T%C3%BCbingen",
    "https://tropter.com/en/germany/tubingen",
    "https://www.ostrichtrails.com/europe/germany/tubingen-walking-tour/",
    "https://www.keeptravel.com/germany/tuebingen",
    "https://www.trip.com/travel-guide/attraction/tubingen-44519/tourist-attractions/",
    "https://www.komoot.com/guide/210692/de-mooiste-attracties-rond-tuebingen",
    "https://www.alltrails.com/trail/germany/baden-wurttemberg/von-boblingen-nach-tubingen",
    "https://www.neuralword.com/en/travel-tourism/europe/exploring-tubingen-must-see-attractions-and-sites",
    "https://pineqone.com/attractions/stadtmuseum-tubingen/",
    "https://www.atlasobscura.com/things-to-do/tubingen-germany",
    "https://www.try-travel.com/category/europe/germany/tubingen/",
    "https://attractions.pw/blog/tubingen/",
    "https://evendo.com/locations/germany/hohenzollern-castle/attraction/st-george-s-collegiate-church-tubingen",
    "https://tripomatic.com/en/list/top-tourist-attractions-in-regierungsbezirk-tubingen-region:13332",
    "www.thecrazytourist.com/15-best-things-reutlingen-germany/",
    "https://freiburger-bote.de/freizeit/freizeitaktivitaeten-tuebingen/",
    "https://www.krone-tuebingen.de/en/a-holiday-in-tuebingen/tuebingen-the-surrounding-area/",
    "https://www.germansights.com/tubingen/",
    "https://www.fodors.com/world/europe/germany/heidelberg-and-the-neckar-valley/places/tubingen/things-to-do/sights",
    "https://www.visitacity.com/en/tubingen/attraction-by-type/all-attractions",
    "https://angiestravelroutes.com/en/12of12-tuebingen-sightseeing/",
    "https://uni-tuebingen.de/en/faculties/faculty-of-economics-and-social-sciences/subjects/department-of-social-sciences/methods-center/events/past-events/fgme2017/tuebingen-1/attractions/",
    "https://visit-tubingen.co.uk/category/things-to-do-in-tubingen/",
    "https://www.outdooractive.com/en/travel-guide/germany/tuebingen/1022199/",
    "https://www.visit-bw.com/en/article/old-town-of-tubingen/c82ceb67-f78d-4b4e-8911-3972ca794cbc#/",
    "https://www.triphobo.com/places/t-bingen-germany/things-to-do",
    "https://uni-tuebingen.de/forschung/zentren-und-institute/brasilien-und-lateinamerika-zentrum/german-brazilian-symposium-2024/about-tuebingen/welcome-to-tuebingen/",
    "https://www.expedia.com/Things-To-Do-In-Tuebingen.d181220.Travel-Guide-Activities",
    "https://triplyzer.com/germany/things-to-do-in-tubingen/",
    "https://explorial.com/uncover-tubingens-hidden-gems-10-lesser-known-spots/",
    "https://www.tripadvisor.de/Restaurants-g198539-Tubingen_Baden_Wurttemberg.html",
    "https://www.tuebingen-info.de/de/mein-aufenthalt/gastronomie/restaurants-und-gaststaetten",
    "https://www.reservix.de/sport-in-tuebingen?_locale=en",
    "https://www.outdooractive.com/en/sports/tuebingen/athletes-destinations-in-tuebingen/21876962/",
    "https://uni-tuebingen.de/en/facilities/central-institutions/university-sports-center/sports-program/",
    "https://uni-tuebingen.de/en/faculties/faculty-of-economics-and-social-sciences/subjects/department-of-social-sciences/sports-science/institute/",
    "https://www.iwm-tuebingen.de/en/research/projects/Vertrauen?name=Vertrauen",

    # Gastronomy & Restaurants
    "https://www.tripadvisor.com/Restaurants-g198539-Tubingen_Baden_Wurttemberg.html",
    "https://guide.michelin.com/en/baden-wurttemberg/tubingen/restaurants",
    "https://www.outdooractive.com/en/eat-and-drink/tuebingen/eat-and-drink-in-tuebingen/21873363/",
    "https://www.opentable.com/food-near-me/stadt-tubingen-germany",
    "https://www.yelp.com/search?find_desc=Restaurants&find_loc=T%C3%BCbingen",
    "https://www.happycow.net/europe/germany/tubingen/",
    "https://www.wurstkueche.com/en/restaurant-our-place/",
    "https://www.1821tuebingen.de/",
    "https://www.lacasa-tuebingen.de/en/index.php",
    "https://www.restaurant-waldhorn.de/en/",
    "https://www.maugan.de/en/",
    "https://en.wikivoyage.org/wiki/T%C3%BCbingen#Eat",
    "https://en.wikivoyage.org/wiki/T%C3%BCbingen#Drink",
    "https://listium.com/@thedesigntourist/98696/15-must-visit-spots-in-tbingen-germany",
    "https://restaurantguru.com/Tubingen#restaurant-list",
    "https://www.kikijourney.com/tuebingen-christmas-market-germany-the-ultimate-guide/",
    "https://www.getyourguide.com/tubingen-l150729/genussvoll-durch-tubingen-kulinarische-stadtfuhrung-t644161/",
    "https://www.faros-tuebingen.com/",
    "https://christmas-events-near-me.com/tuebingen-germany/",
    "https://www.my-stuwe.de/en/refectory/",
    "https://www.eventbrite.com/b/germany--t%C3%BCbingen/food-and-drink/",
    "https://wanderlog.com/list/geoCategory/199488/where-to-eat-best-restaurants-in-tubingen",
    "https://wanderlog.com/list/geoCategory/312176/best-spots-for-lunch-in-tubingen",
    "https://www.tripadvisor.com/Restaurants-g198539-Tubingen_Baden_Wurttemberg.html",
    "https://www.tripadvisor.com/Restaurants-g198539-zfp58-Tubingen_Baden_Wurttemberg.html",
    "https://guide.michelin.com/en/baden-wurttemberg/tubingen/restaurants",
    "https://guide.michelin.com/us/en/baden-wurttemberg/tbingen/restaurants",
    "https://www.outdooractive.com/en/eat-and-drink/tuebingen/eat-and-drink-in-tuebingen/21873363/",
    "https://www.opentable.com/food-near-me/stadt-tubingen-germany",
    "https://www.yelp.com/search?find_desc=Restaurants&find_loc=T%C3%BCbingen",
    "https://m.yelp.com/search?cflt=swabian&find_loc=T%C3%BCbingen%2C+Baden-W%C3%BCrttemberg",
    "https://www.happycow.net/europe/germany/tubingen/",
    "https://www.wurstkueche.com/en/restaurant-our-place/",
    "https://www.1821tuebingen.de/",
    "https://www.lacasa-tuebingen.de/en/index.php",
    "https://www.restaurant-waldhorn.de/en/",
    "https://www.maugan.de/en/",
    "https://en.wikivoyage.org/wiki/T%C3%BCbingen#Eat",
    "https://en.wikivoyage.org/wiki/T%C3%BCbingen#Drink",
    "https://listium.com/@thedesigntourist/98696/15-must-visit-spots-in-tbingen-germany",
    "https://www.tripadvisor.com/Restaurants-g198539-Tubingen_Baden_Wurttemberg.html",
    "https://restaurantguru.com/Tubingen#restaurant-list",
    "https://www.kikijourney.com/tuebingen-christmas-market-germany-the-ultimate-guide/",
    "https://www.getyourguide.com/tubingen-l150729/genussvoll-durch-tubingen-kulinarische-stadtfuhrung-t644161/",
    "https://www.faros-tuebingen.com/",
    "https://christmas-events-near-me.com/tuebingen-germany/",
    "https://uni-tuebingen.de/en/international/welcome-center/registration/",
    "https://www.my-stuwe.de/en/refectory/",
    "https://www.opentable.com/food-near-me/stadt-tubingen-germany",
    "https://www.eventbrite.com/b/germany--t%C3%BCbingen/food-and-drink/",
    "https://www.yelp.com/search?find_desc=Drink&find_loc=T%C3%BCbingen%2C+Baden-W%C3%BCrttemberg",
    "https://www.ubereats.com/de-en/city/t%C3%BCbingen-bw?srsltid=AfmBOooVHU1cR4Js1VRCXuua2RVKE36M4jfq-FmEbFDbrMJojWYBNwSY",
    "https://www.my-stuwe.de/en/refectory/cafeteria-unibibliothek-tuebingen/",
    "https://www.tripadvisor.com/Restaurant_Review-g198539-d3193708-Reviews-Kuckuck-Tubingen_Baden_Wurttemberg.html",
    "https://www.wurstkueche.com/en/restaurant-our-place/",
    "https://www.mph.tuebingen.mpg.de/canteen",
    "https://www.reddit.com/r/Tuebingen/comments/10s7frj/best_burger_place_in_tubingen/",
    "https://neckawa.de/wp-content/uploads/2024/07/2024_07_05-Speisekarten_Neckawa-web.pdf",
    "https://uni-tuebingen.de/en/international/sprachen-lernen/deutschkurse/suedafrika-programm/programm-2017/tuebingen-diary/german-food/",
    "https://www.krone-tuebingen.de/en/restaurants/restaurant-ludwigs/",
    "https://www.outdooractive.com/en/places-to-eat-and-drink/tuebingen/eat-and-drink-in-tuebingen/21873363/",
    "https://uni-tuebingen.de/fakultaeten/philosophische-fakultaet/fachbereiche/geschichtswissenschaft/seminareinstitute/neuere-geschichte/studium/lehrveranstaltungen/civis-classes/",
    "https://www.lieferando.de/en/delivery/food/tuebingen-72072"

    # Local News & Community (English)
    "https://integreat.app/tuebingen/en/news/tu-news",
    "https://tunewsinternational.com/category/news-in-english/",
    "https://www.reddit.com/r/Tuebingen/",
    "https://www.reddit.com/r/Tuebingen/comments/1rhpyk/life_as_an_english_speaking_person_in_t%C3%BCbingen/",
    "https://www.meetup.com/tubingen-meet-mingle/",
    "https://www.internations.org/germany-expats/",
    "https://www.facebook.com/groups/tuebingenexpats/",
    "https://www.couchsurfing.com/places/europe/germany/tubingen",

    # Accommodation
    "https://www.booking.com/city/de/tubingen.html",
    "https://all.accor.com/a/en/destination/city/hotels-tubingen-v4084.html",
    "https://www.hotel-am-schloss.de/en/",
    "https://www.ibis.com/gb/hotel-3200-ibis-tuebingen/index.shtml",
    "https://www.kronprinz-tuebingen.de/en/",

    # Culture & Arts / Events
    "https://www.stadtmuseum-tuebingen.de/english/",
    "https://www.kunsthallentuebingen.de/en/",
    "https://www.landestheater-tuebingen.de/en/home/",
    "https://www.franzk.net/en/",
    "https://www.kino-arsenal.de/",
    "https://www.tuebinger-kultursommer.de/",
    "https://www.tuebinger-stadtlauf.de/en/home/",
    "https://rausgegangen.de/en/tubingen/",
    "https://www.bandsintown.com/c/tuebingen-germany",
    "https://www.eventbrite.com/d/germany--t%C3%BCbingen/english/",

    # Outdoor & Recreation
    "https://www.alltrails.com/germany/baden-wurttemberg/tubingen",
    "https://www.outdooractive.com/en/city-walks/tuebingen/city-walks-in-tuebingen/8232815/",
    "https://www.outdooractive.com/en/hiking-trails/tuebingen/hiking-in-tuebingen/1432855/",
    "https://www.komoot.com/guide/881/hiking-around-landkreis-tuebingen",
    "https://www.wikiloc.com/trails/hiking/germany/baden-wurttemberg/tubingen",
    "https://www.outdooractive.com/en/travel-guide/germany/tuebingen/1022199/",

    # Sports & Recreation
    "https://www.reservix.de/sport-in-tuebingen?_locale=en",
    "https://www.outdooractive.com/en/sports/tuebingen/athletes-destinations-in-tuebingen/21876962/",
    "https://uni-tuebingen.de/en/facilities/central-institutions/university-sports-center/",
    "https://uni-tuebingen.de/en/facilities/central-institutions/university-sports-center/sports-program/",
    "https://www.sportzentrum.uni-tuebingen.de/en/",
    "https://www.outdooractive.com/en/swimming-pools/tuebingen/swimming-in-tuebingen/21875473/",
    "https://www.outdooractive.com/en/fitness-centres/tuebingen/fitness-centres-in-tuebingen/21875484/",

    # Business & Local Economy
    "https://www.hk-reutlingen.de/en/start/",

    # Shopping & Commerce
    "https://tuebingenresearchcampus.com/en/tuebingen/living-in-tuebingen/shopping",
    "https://www.tripadvisor.com/Attractions-g198539-Activities-c26-Tubingen_Baden_Wurttemberg.html",
    "https://cityseeker.com/tuebingen-de/shopping",
    "https://www.outdooractive.com/en/shoppings/tuebingen/shopping-in-tuebingen/21876964/",
    "https://www.yelp.com/search?cflt=shopping&find_loc=T%C3%BCbingen%2C+Baden-W%C3%BCrttemberg",
    "https://uni-tuebingen.de/international/studierende-aus-dem-ausland/erasmus-und-austausch-nach-tuebingen/studentisches-leben/tuebingen-basics-and-beyond/living-in-tuebingen/",
    "https://uni-tuebingen.de/en/280707",
    "https://us.trip.com/travel-guide/shops/city-44519/",

    # Transportation & Mobility
    "https://tuebingenresearchcampus.com/en/tuebingen/living-in-tuebingen/mobility/by-public-transport",
    "https://www.bahnhof.de/en/tuebingen-hbf/parking-spaces",
    "https://www.swtue.de/en/private-customer/parking.html",
    "https://www.tripadvisor.com/ShowTopic-g198539-i4335-k5492275-How_can_I_use_the_bus_lines-Tubingen_Baden_Wurttemberg.html",
    "https://tuebingen.ai/wiki/transportation-and-mobility",
    "https://www.bahnhof.de/en/tuebingen-hbf/map",
    "https://uni-tuebingen.de/en/university/how-to-get-here/",
    "https://en.parkopedia.com/parking/t%C3%BCbingen/",
    "https://www.tuebingen-info.de/en/service/vor-ort/mit-dem-auto",

    # Health & Medical Services
    "https://www.uniklinik-tuebingen.de/en/",
    "https://tuebingenresearchcampus.com/en/tuebingen/living-in-tuebingen/healthcare",
    "https://uni-tuebingen.de/en/280706",
    "https://www.doctolib.de/city/tuebingen",
    "https://www.kliniksuche.de/city/tuebingen",

    # Real Estate & Housing
    "https://www.studenten-wg.de/city/tuebingen",
    "https://www.wg-gesucht.de/en/wg-zimmer-in-Tuebingen.125.0.1.0.html",
    "https://www.immobilienscout24.de/Suche/de/baden-wuerttemberg/tuebingen",
    "https://tuebingenresearchcampus.com/en/tuebingen/living-in-tuebingen/housing",
    "https://www.welcome.uni-tuebingen.de/housing/",
    "https://uni-tuebingen.de/en/study/student-life/student-housing/",
    "https://uni-tuebingen.de/en/international/study-in-tuebingen/degree-seeking-students/getting-started-and-orientation-for-international-students/housing/",
    "https://www.my-stuwe.de/en/housing/halls-of-residence-tuebingen/",
    "https://tuebingenresearchcampus.com/en/tuebingen/accommodations/finding-a-home/housing-options",

    # Education (Non-University)
    "https://www.goethe.de/ins/de/en/sta/tue.html",
    "https://www.berlitz.com/locations/germany/tuebingen",
    "https://tuebingenresearchcampus.com/en/tuebingen/living-in-tuebingen/language-courses",

    # Services & Utilities
    "https://tuebingenresearchcampus.com/en/tuebingen/living-in-tuebingen/banking",
    "https://tuebingenresearchcampus.com/en/tuebingen/living-in-tuebingen/internet-and-mobile",
    "https://www.swtue.de/en/",
    "https://www.deutsche-post.de/en/branch-finder.html",

    # Student Life & University Services
    "https://uni-tuebingen.de/en/study/student-life/",
    "https://www.my-stuwe.de/en/refectory/cafeteria-unibibliothek-tuebingen/",
    "https://www.mph.tuebingen.mpg.de/canteen",
    "https://uni-tuebingen.de/en/international/welcome-center/registration/",

    # Forums
    "https://uni-tuebingen.de/forschung/zentren-und-institute/tuebinger-forum-fuer-wissenschaftskulturen-tfw/tfw/",
    "https://uni-tuebingen.de/forschung/zentren-und-institute/forum-scientiarum/",
    "https://www.studis-online.de/Fragen-Brett/list.php?84",
    "https://stadtistik.de/stadt/tuebingen-08416041/",
    "https://www.juristischesforum.de/",
    "https://www.studis-online.de/Fragen-Brett/read.php?84,701326,1969302,quote=1",
    "https://www.tuebingen.de/Dateien/Praesentation5Altstadtforum.pdf",
    "https://uni-tuebingen.de/en/excellence-strategy/research/platforms/global-encounters/tuebingen-forum-on-social-resonances-of-societal-crises-tueforce/",
    "https://erasmusu.com/en/erasmus-tubingen/erasmus-forum",
    "https://www.rosalux.de/en/publication/id/2539/1-volxuni-des-social-forum-tuebingen-reutlingen",
    "https://www.kulturforum.info/de/forum-partner/alle-partner/6016-1001121-institut-fuer-osteuropaeische-geschichte-und-landeskunde-an-der-eberhard-karls-universitaet-tuebingen",
    "https://www.tripadvisor.com/ShowForum-g198539-i4335-Tubingen_Baden_Wurttemberg.html",
    "https://uni-tuebingen.de/en/research/centers-and-institutes/tuebingen-forum-for-science-and-humanities/tfw/",
    "https://www.reddit.com/r/Tuebingen/",
    "https://ps-forum.cs.uni-tuebingen.de/login",
    "https://www.reddit.com/r/UniTuebingen/",
    "https://uni-tuebingen.de/en/research/centers-and-institutes/tuebingen-forum-for-science-and-humanities/about/institution/",
    "https://www.facebook.com/FreigeistForumTubingen/posts/hier-gesammelte-vortr%C3%A4ge-und-interviewshttpwwwfreigeist-forum-tuebingendepvortra/815537238534598/",
    "https://community.ricksteves.com/travel-forum/germany-reviews/tubingen",
    "https://www.n-k-t.de/"
]
                    


def init_search():
    indexer = Indexer()
    hybrid_model = HybridRetrieval()
    sentiment_pipeline = pipeline("text-classification", model="GroNLP/mdebertav3-subjectivity-english", device=-1)

    return indexer, hybrid_model, sentiment_pipeline