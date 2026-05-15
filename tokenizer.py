from talasalitaan import Talasalitaan
import json

TRAIN_MODE = False

with open("settings.json", 'r', encoding="utf-8") as f:
    FOLDER_PATH = json.load(f)

tokenizer = Talasalitaan()

if TRAIN_MODE:
    tokenizer.train(FOLDER_PATH["path/to/corpus"])

sp = tokenizer.spm_instance

texts = [
    "Kamusta, mga kababayan!", 
    "pagpapanibagong-tatag", 
    "nakakapagpabagabag",
    "pinakanakapagpapabagabag",
    "Magandang umaga, kapatid!",
    "Kumain siya ng pagkain.",
    "kumain ka na ba?",
    "Ito ay pagsasalaysay ng mga katutubo sa kanilang paniniwalang lakas ng pisikal na kapaligiran at lakas ng pananampalataya ng lumilimbag sa kanilang buhay at kapalaran.",
    "Ang unang paglalayag na pambuong mundo sa ngalan ng Espanya ay nasundan ng apat pang mga ekspedisyon mula 1525 hanggang 1542. Sa ikaapat na panggagalugad, narating ni Ruy Lopez de Villalobos ang Kapuluan ng Pilipinas at pinangalanan niya ang mga pulo mula kay Philip II na noon ay may katayuan bilang tagapagmana ng trono ng Kaharian ng Espanya, bagaman hindi pa pormal na naitatag ang Pilipinas bilang opisyal na teritoryo ng Espanya.",
    "Sapagkat ang pagkilala sa katutubong karangalan at sa pantay at di-maikakait na mga karapatan ng lahat ng nabibilang sa angkan ng tao ay siyang saligan ng kalayaan, katarungan at kapayapaan sa daigdig.",
    "Ang Wikipedia ay isang proyektong online na ensiklopedya na panlahat, nakasulat sa maraming wika, at pinagtutulungan ang paggawa ng mga artikulo sa prinsipyong wiki. Naglalayon ang proyektong ito na mag-alok ng mga nilalaman na malayang muling magagamit, walang pinapanigan, at napapatunayan, na maaring baguhin at mapabuti ninuman. Nakikilala ang Wikipedia sa pamamagitan ng mga naitatag na prinsipyo. Nakalisensiya ang nilalaman nito sa ilalim ng Creative Commons BY-SA. Maari itong kopyahin at muling gamitin sa ilalim ng parehong lisensiya, na sumasailalim sa paggalang sa mga kondisyon. Ibinbigay ng Wikipedia ang mga nilalaman nito ng walang bayad, walang patalastas, at hindi nagsasamantala sa paggamit ng personal na datos ng mga gumagamit nito.",
    
    # From (as example): https://www.tagaloglang.com/talumpati-ni-nelson-mandela/
    "Ang ating mga nagawa bilang ordinaryong mamamayan ng Timog Africa ay kailangang magbunga ng tunay na mamamayan nito na magpapalawak sa paniniwala ng sangkatauhan sa katarungan, magpapalakas sa tiwala sa kadakilaan ng kaluluwa, at magtutustos sa lahat ng ating pag-asa sa kapakinabangan ng buhay ng lahat.",
    
    # From (as example): https://www.gmanetwork.com/news/balitambayan/umg/987386/drawer-ng-cabinet-minulto-nga-ba-matapos-na-mahuli-cam-na-nagbukas-sara/story/ 
    "Nabalot ng kababalaghan ang masaya sanang bonding ng magkakaibigan nang bigla na lang magbukas-sara na mag-isa sa kanilang harapan ang drawer ng isang cabinet. Ang kinaroroonan ng cabinet, isang bahay-bakasyunan na pinaparentahan at kamamatay lang umano ng may-ari.",
    
    # From (as example): https://bandera.inquirer.net/444456/bini-jhoanna-kinabog-weather-report-sa-good-day-la-achieve-sa-bucket-list
    "NATUPAD ang isa sa bucket list ng BINI leader na si Jhoanna Robles, habang nasa Amerika. Biglaan kasi siyang naging weather presenter nang mag-guest ang nation's girl group sa morning show na Good Day LA, kung saan una nilang ibinahagi ang kanilang makasaysayang performance sa Coachella, pati na rin ang kanilang bagong EP na \"Signals\" at nalalapit na world tour. Pero imbes na matapos lang sa chikahan, biglang nagkaroon ng nakakatuwang twist!"
]


for text in texts:
    tokens = sp.encode(text, out_type=str)
    ids = sp.encode(text, out_type=int)

    print(tokens)
    print(f"Length: {len(ids)}")

    # Decode back
    decoded = sp.decode(ids)
    print(decoded)
    print()