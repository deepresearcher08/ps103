"""District/city-level geocoding for MoSPI project names.

The Flash Report names embed the PROJECT LOCATION (city, district HQ, town) in
the plain-English name — e.g. "CONSTRUCTION OF DOMESTIC PASSENGER TERMINAL
BUILDING AT DEHRADUN AIRPORT, DEHRADUN", "PATNA GAYA DOBHI NH 139", "KUDANKULAM
NUCLEAR POWER PROJECT". We match those tokens against this table of town /
district-centroid coordinates to replace STATE-CENTROID geo (which is a
categorical state ID wearing a continuous mask) with genuine within-state
spatial variation.

Honesty notes (for the demo / pitch):
  - coordinates are town/district-centroid level, NOT street-level geocoded
    addresses; where a name has no city token we keep the state centroid.
  - the COAL-ORG map (BCCL -> Dhanbad, NCL -> Singrauli, ...) uses each
    public-sector coal company's OPERATING AREA (a district), which is public
    knowledge — a defensible district-level location for mine/site names that
    carry no town token.
"""
from __future__ import annotations

import math
import re
from typing import Optional

# --------------------------------------------------------------------------- #
# town / district-centroid database:  NORMALIZED name -> (lat, lon, display)
# Keys are canonical UPPERCASE; matching strips spaces so multi-word towns
# (COOCH BEHAR, TIRUCHCHIRAPPALLI, ...) and PDF-mangled names both resolve.
# --------------------------------------------------------------------------- #
_DISTRICTS: dict[str, tuple[float, float, str]] = {
    # -- megacities / metro --
    "MUMBAI": (19.0760, 72.8777, "MUMBAI"),
    "NAVI MUMBAI": (19.0330, 73.0297, "NAVI MUMBAI"),
    "POWAI": (19.1176, 72.9060, "POWAI"),
    "DELHI": (28.6139, 77.2090, "DELHI"),
    "NEW DELHI": (28.6139, 77.2090, "NEW DELHI"),
    "KOLKATA": (22.5726, 88.3639, "KOLKATA"),
    "CHENNAI": (13.0827, 80.2707, "CHENNAI"),
    "BANGALORE": (12.9716, 77.5946, "BANGALORE"),
    "BENGALURU": (12.9716, 77.5946, "BENGALURU"),
    "HYDERABAD": (17.3850, 78.4867, "HYDERABAD"),
    "PUNE": (18.5204, 73.8567, "PUNE"),
    "AHMEDABAD": (23.0225, 72.5714, "AHMEDABAD"),
    "SURAT": (21.1702, 72.8311, "SURAT"),
    "JAIPUR": (26.9124, 75.7873, "JAIPUR"),
    "KANPUR": (26.4499, 80.3319, "KANPUR"),
    "LUCKNOW": (26.8467, 80.9462, "LUCKNOW"),
    "NAGPUR": (21.1458, 79.0882, "NAGPUR"),
    "INDORE": (22.7196, 75.8577, "INDORE"),
    "BHOPAL": (23.2599, 77.4126, "BHOPAL"),
    "KOCHI": (9.9312, 76.2673, "KOCHI"),
    "THIRUVANANTHAPURAM": (8.5241, 76.9366, "THIRUVANANTHAPURAM"),
    "KOZHIKODE": (11.2588, 75.7804, "KOZHIKODE"),
    "CHANDIGARH": (30.7333, 76.7794, "CHANDIGARH"),
    "BHUBANESWAR": (20.2961, 85.8245, "BHUBANESWAR"),
    "PATNA": (25.5941, 85.1376, "PATNA"),
    "RANCHI": (23.3441, 85.3096, "RANCHI"),
    "GUWAHATI": (26.1445, 91.7362, "GUWAHATI"),
    "DEHRADUN": (30.3165, 78.0322, "DEHRADUN"),
    "SHIMLA": (31.1048, 77.1734, "SHIMLA"),
    "RAIPUR": (21.2514, 81.6296, "RAIPUR"),
    "JAMMU": (32.7266, 74.8570, "JAMMU"),
    "SRINAGAR": (34.0837, 74.7973, "SRINAGAR"),
    "AGARTALA": (23.8315, 91.2868, "AGARTALA"),
    "AIZAWL": (23.7271, 92.7176, "AIZAWL"),
    "SHILLONG": (25.5788, 91.8933, "SHILLONG"),
    "IMPHAL": (24.8170, 93.9368, "IMPHAL"),
    "KOHIMA": (25.6744, 94.1086, "KOHIMA"),
    "DIMAPUR": (25.8934, 93.9271, "DIMAPUR"),
    "PORT BLAIR": (11.6234, 92.7265, "PORT BLAIR"),
    "PORTBLAIR": (11.6234, 92.7265, "PORTBLAIR"),
    "PONDICHERRY": (11.9416, 79.8083, "PONDICHERRY"),
    "PUDUCHERRY": (11.9416, 79.8083, "PUDUCHERRY"),
    # -- airports / AAI --
    "JABALPUR": (23.1815, 79.9864, "JABALPUR"),
    "HISAR": (29.1492, 75.7217, "HISSAR"),
    "HISSAR": (29.1492, 75.7217, "HISSAR"),
    "RAJKOT": (22.3039, 70.8022, "RAJKOT"),
    "AYODHYA": (26.7986, 82.1996, "AYODHYA"),
    "GWALIOR": (26.2183, 78.1828, "GWALIOR"),
    "TUTICORIN": (8.7642, 78.1348, "TUTICORIN"),
    "THOOTHUKUDI": (8.7642, 78.1348, "TUTICORIN"),
    "ITANAGAR": (27.0844, 93.6053, "ITANAGAR"),
    "HOLLONGI": (27.09, 93.68, "HOLLONGI"),
    "LEH": (34.1526, 77.5771, "LEH"),
    "VIJAYAWADA": (16.5062, 80.6480, "VIJAYAWADA"),
    "PRAYAGRAJ": (25.4358, 81.8463, "PRAYAGRAJ"),
    "TIRUCHIRAPALLI": (10.7905, 78.7047, "TIRUCHIRAPALLI"),
    "TIRUCHCHIRAPPALLI": (10.7905, 78.7047, "TIRUCHIRAPALLI"),
    "TRICHY": (10.7905, 78.7047, "TIRUCHIRAPALLI"),
    "SAFDARJUNG": (28.5827, 77.2109, "DELHI"),
    "NAGPUR AIRPORT": (21.0924, 79.0472, "NAGPUR"),
    # -- nuclear / thermal / power --
    "KUDANKULAM": (8.1739, 77.7108, "KUDANKULAM"),
    "KAKRAPAR": (21.6150, 73.0450, "KAKRAPAR"),
    "BHAVINI": (12.5567, 80.1756, "KALPAKKAM"),
    "RAWATBHATA": (24.82, 75.78, "RAWATBHATA"),
    "NEYVELI": (11.5368, 79.4474, "NEYVELI"),
    "SINGRAULI": (24.1983, 82.6445, "SINGRAULI"),
    "KORBA": (22.3594, 82.6794, "KORBA"),
    "SIKAR": (27.6094, 75.1399, "SIKAR"),
    "JODHPUR": (26.2389, 73.0243, "JODHPUR"),
    "BARMER": (25.7485, 71.3935, "BARMER"),
    "SATARA": (17.6863, 74.0189, "SATARA"),
    "BHUSAWAL": (21.0437, 75.8024, "BHUSAWAL"),
    "CHANDRAPUR": (19.9615, 79.2961, "CHANDRAPUR"),
    "PARLI": (18.8511, 76.5324, "PARLI"),
    "GANDHINAGAR": (23.2156, 72.6369, "GANDHINAGAR"),
    "MUNDRA": (22.8389, 69.7117, "MUNDRA"),
    # -- coal fields (org operating areas used as fallback, and town hits) --
    "DHANBAD": (23.7957, 86.4304, "DHANBAD"),
    "PATHERDIH": (23.72, 86.40, "DHANBAD COALFIELD"),
    "BHOJUDIH": (23.70, 86.45, "DHANBAD COALFIELD"),
    "MOONIDIH": (23.75, 86.50, "DHANBAD COALFIELD"),
    "MURAIDIH": (23.74, 86.47, "DHANBAD COALFIELD"),
    "MADHUBAND": (23.71, 86.41, "DHANBAD COALFIELD"),
    "CHURI": (23.78, 86.42, "DHANBAD COALFIELD"),
    "TETARIAKHAR": (23.72, 86.44, "DHANBAD COALFIELD"),
    "RANIGANJ": (23.6161, 87.1280, "RANIGANJ"),
    "ASANSOL": (23.6830, 86.9710, "ASANSOL"),
    "DURGAPUR": (23.5204, 87.3119, "DURGAPUR"),
    "HURA": (23.65, 87.03, "RANIGANJ COALFIELD"),
    "ANGUL": (20.8409, 85.0930, "ANGUL"),
    "TALCHER": (20.9516, 85.2336, "TALCHER"),
    "GARJANBAHAL": (20.84, 85.01, "ANGUL COALFIELD"),
    "KULDA": (20.86, 85.05, "ANGUL COALFIELD"),
    "BASUNDHARA": (20.82, 85.10, "ANGUL COALFIELD"),
    "BONJEMEHARI": (20.87, 85.08, "ANGUL COALFIELD"),
    "MOHANPUR": (23.63, 87.16, "MAJHIA COALFIELD"),
    "SHYAMSUNDARPUR": (23.64, 87.10, "RANIGANJ COALFIELD"),
    "TILABONI": (23.85, 87.13, "RANIGANJ COALFIELD"),
    "PARASEA": (23.61, 87.09, "RANIGANJ COALFIELD"),
    "KAMARJANGUL": (23.64, 87.06, "RANIGANJ COALFIELD"),
    "DUDHICHUA": (24.20, 82.62, "SINGRAULI COALFIELD"),
    "BINA": (24.1719, 78.1937, "BINA"),
    "KAKRI": (24.13, 82.74, "SINGRAULI COALFIELD"),
    "KHADIA": (24.05, 82.65, "SINGRAULI COALFIELD"),
    "GODAVARIKHANI": (18.81, 79.51, "GODAVARIKHANI"),
    "KAKATIYAKHANI": (18.73, 79.33, "PALONCHA COALFIELD"),
    "KOTHAGUDEM": (17.5546, 80.6208, "KOTHAGUDEM"),
    "RAMAGUNDAM": (18.7580, 79.4733, "RAMAGUNDAM"),
    "MANCHERIAL": (18.8730, 79.4430, "MANCHERIAL"),
    "INDARAM": (18.87, 79.63, "INDARAM COAL BLOCK"),
    "KARMA": (23.47, 85.49, "CCL NORTH KARANPURA"),
    "RAJRAPPA": (23.4825, 85.5617, "RAJRAPPA"),
    "HAZARIBAGH": (23.9925, 85.3637, "HAZARIBAGH"),
    "BOKARO": (23.6693, 86.1511, "BOKARO"),
    "KIRANDUL": (18.65, 81.26, "KIRANDUL"),
    "DANTEWADA": (18.9038, 81.3407, "DANTEWADA"),
    "BILASPUR": (22.0797, 82.1390, "BILASPUR"),
    "CHITRA": (23.87, 86.20, "CHITRA (BOKARO COALFIELD)"),
    # -- refineries / petroleum / pipelines --
    "GUWAHATI REFINERY": (26.16, 91.70, "GUWAHATI REFINERY"),
    "DIGBOI": (27.3935, 95.6220, "DIGBOI"),
    "NUMALIGARH": (26.61, 93.51, "NUMALIGARH"),
    "VISAKHAPATNAM": (17.6868, 83.2185, "VISAKHAPATNAM"),
    "VISAKH": (17.6868, 83.2185, "VISAKHAPATNAM"),
    "DHARMAPURI": (12.1277, 78.1577, "DHARMAPURI"),
    "MANGALORE": (12.9141, 74.8560, "MANGALORE"),
    "KOOTTANAD": (10.10, 76.25, "KOOTTANAD"),
    "COIMBATORE": (11.0168, 76.9558, "COIMBATORE"),
    "IRUGUR": (11.04, 76.96, "IRUGUR (COIMBATORE)"),
    "MADURAVOYAL": (13.07, 80.15, "MADURAVOYAL"),
    "HALDIA": (22.0605, 88.1164, "HALDIA"),
    "BARAUAL": (25.90, 84.89, "BARAUNI"),
    "BARAUND": (25.95, 85.05, "BARAUNI"),
    "HALDIA PIPELINE": (22.06, 88.12, "HALDIA"),
    "KANDLA": (23.0333, 70.2167, "KANDLA"),
    # -- rail doubling / new lines --
    "KATRA": (33.0244, 74.9310, "KATRA"),
    "AMRITSAR": (31.6340, 74.8723, "AMRITSAR"),
    "BATHINDA": (30.2110, 74.9379, "BATHINDA"),
    "SAHARSA": (25.8850, 86.6000, "SAHARSA"),
    "PURNEA": (25.7824, 87.4738, "PURNEA"),
    "PURNIA": (25.7824, 87.4738, "PURNEA"),
    "ARARIA": (26.1440, 87.5130, "ARARIA"),
    "GALGALIA": (26.19, 87.30, "GALGALIA"),
    "BALURGHAT": (25.2160, 88.7730, "BALURGHAT"),
    "MAU": (25.9417, 83.5612, "MAU"),
    "INDARA": (25.97, 83.53, "INDARA (MAU)"),
    "GORAKHPUR": (26.7606, 83.3732, "GORAKHPUR"),
    "AMETHI": (26.7640, 81.2020, "AMETHI"),
    "SAMBALPUR": (21.4719, 83.9830, "SAMBALPUR"),
    "SAMABALPUR": (21.4719, 83.9830, "SAMBALPUR"),
    "DHANBAD RAIL": (23.80, 86.43, "DHANBAD"),
    "KOTSHILA": (23.72, 86.06, "KOTSHILA"),
    "NAGORE": (10.8160, 79.8490, "NAGORE"),
    "KARAIKAL": (10.9251, 79.8385, "KARAIKAL"),
    "ETAHWAH": (26.7750, 79.0300, "ETAWAH"),
    "ETAWAH": (26.7750, 79.0300, "ETAWAH"),
    "HISUA": (25.0400, 85.4100, "HISUA"),
    "RAJGIR": (25.0250, 85.4162, "RAJGIR"),
    "RAJGIRI": (25.0250, 85.4162, "RAJGIR"),
    "MOONAK": (29.7900, 75.7200, "MOONAK (HISAR)"),
    "JAKHAL": (29.8000, 75.6100, "JAKHAL"),
    "BUDHLADA": (30.0300, 75.5500, "BUDHLADA"),
    "JOKA": (22.4600, 88.3000, "JOKA (KOLKATA)"),
    "SAIRONG": (24.5600, 94.3000, "SAIRONG"),
    "PISKA": (23.55, 85.25, "PISKA (RANCHI)"),
    "SAGAULI": (26.80, 84.75, "SAGAULI (EAST CHAMPARAN)"),
    "BHAGALPUR": (25.2425, 86.9872, "BHAGALPUR"),
    "MIRZACHAUKI": (25.24, 86.98, "MIRZACHAUKI (BHAGALPUR)"),
    # -- highways / expressways / roads --
    "DWARKA": (28.5923, 77.0460, "DWARKA (DELHI)"),
    "PANIPAT": (29.3909, 76.9635, "PANIPAT"),
    "SHAMLI": (29.4488, 77.3120, "SHAMLI"),
    "MUZAFFARNAGAR": (29.4727, 77.7080, "MUZAFFARNAGAR"),
    "PILKHUWA": (28.7100, 77.2800, "PILKHUWA"),
    "VADODARA": (22.3072, 73.1812, "VADODARA"),
    "VARANASI": (25.3176, 82.9739, "VARANASI"),
    "RAMPUR": (28.8137, 79.0292, "RAMPUR"),
    "KATHGODAM": (29.2665, 79.5448, "KATHGODAM"),
    "GAYA": (24.7914, 85.0002, "GAYA"),
    "DOBHI": (24.9648, 84.4874, "DOBHI"),
    "HARDA": (22.3522, 77.0951, "HARDA"),
    "BETUL": (21.9053, 77.9026, "BETUL"),
    "GHAZIPUR": (25.5833, 83.5850, "GHAZIPUR"),
    "BALLIA": (25.7600, 84.1500, "BALLIA"),
    "CHANDIKHOL": (20.79, 86.06, "CHANDIKHOL (JAJPUR)"),
    "PARADIP": (20.3160, 86.6241, "PARADIP"),
    "HAMIRPUR": (25.9549, 80.1480, "HAMIRPUR"),
    "SAHARANPUR": (29.9640, 77.5460, "SAHARANPUR"),
    "KHAMMAM": (17.2473, 80.1514, "KHAMMAM"),
    "DEVARAPALLE": (17.03, 81.15, "DEVARAPALLE"),
    "MUNGER": (25.3776, 86.4740, "MUNGER"),
    "VALANCHERY": (10.9794, 75.9910, "VALANCHERY"),
    "PARAVUR": (9.9865, 76.2420, "NORTH PARAVUR"),
    "HOSPET": (15.2685, 76.3880, "HOSPET"),
    "BELLARY": (15.1394, 76.9214, "BELLARY"),
    "DHULE": (20.9042, 74.7749, "DHULE"),
    "KOLHAPUR": (16.7050, 74.2433, "KOLHAPUR"),
    "RATNAGIRI": (16.9902, 73.3120, "RATNAGIRI"),
    "PANAGARH": (23.45, 87.43, "PANAGARH"),
    "BARWA": (23.41, 87.45, "BARWA (PANAGARH)"),
    "UDHAMPUR": (32.9301, 75.1350, "UDHAMPUR"),
    "RAMBAN": (33.24, 75.25, "RAMBAN"),
    "HARIDWAR": (29.9457, 78.1642, "HARIDWAR"),
    "NAGINA": (29.4500, 78.4400, "NAGINA"),
    "NERCHOWK": (31.60, 76.67, "NERCHOWK (MANDI)"),
    "PANDOH": (31.55, 76.65, "PANDOH (MANDI)"),
    "SATNA": (24.6005, 80.8322, "SATNA"),
    "REWA": (24.5362, 81.3037, "REWA"),
    "JHABUA": (22.7662, 74.5908, "JHABUA"),
    "SARDARPUR": (22.68, 74.75, "SARDARPUR"),
    "BODHRE": (20.87, 74.72, "BODHRE (DHULE)"),
    "KANCHANPUR": (23.29, 91.29, "KANCHANPUR"),
    "SIMARIA": (25.51, 85.26, "SIMARIA (PATNA)"),
    "DIBRUGARH": (27.4728, 94.9120, "DIBRUGARH"),
    "MORAN": (27.18, 94.93, "MORAN (DIBRUGARH)"),
    "KATHKATI": (26.04, 93.89, "KATHKATI (KARBI ANGLONG)"),
    "PHULBARI": (26.22, 92.07, "PHULBARI (BONGAIGAON)"),
    "CHURAIBARI": (23.78, 91.57, "CHURAIBARI (TRIPURA)"),
    "GAURIPUR": (26.37, 90.03, "GAURIPUR (ASSAM)"),
    "BARPETA": (26.3244, 91.0070, "BARPETA"),
    "SUTARKANDI": (23.78, 91.58, "SUTARKANDI (TRIPURA)"),
    "KAMARDANGA": (26.30, 91.50, "KAMARDANGA (BAKSA)"),
    "DHUBRI": (26.0230, 89.9889, "DHUBRI"),
    "SRIRAMPUR": (26.34, 90.25, "SRIRAMPUR (ASSAM)"),
    "JIRIBAM": (24.8130, 93.1290, "JIRIBAM"),
    "UKHRUL": (25.0964, 94.3614, "UKHRUL"),
    "TOLOI": (25.29, 94.25, "TOLOI (UKHRUL)"),
    "TADUBI": (25.36, 94.19, "TADUBI (UKHRUL)"),
    "MAHUR": (25.17, 93.11, "MAHUR (DIMA HASAO)"),
    "TAMENGLONG": (24.8956, 93.5030, "TAMENGLONG"),
    "CHURACHANDPUR": (24.3276, 93.6729, "CHURACHANDPUR"),
    "TUIVAI": (24.26, 93.73, "TUIVAI (CHURACHANDPUR)"),
    "KHONGSANG": (24.86, 94.20, "KHONGSANG (IMPHAL EAST)"),
    "VAIRENGTE": (23.60, 93.25, "VAIRENGTE (MIZORAM)"),
    "SAIRANG": (23.40, 93.05, "SAIRANG (MIZORAM)"),
    "LUNGLEI": (22.8890, 92.7380, "LUNGLEI"),
    "TLABUNG": (22.56, 92.88, "TLABUNG (LUNGLEI)"),
    "KHOWAI": (23.8900, 91.5960, "KHOWAI"),
    "JOLAIBARI": (23.98, 91.91, "JOLAIBARI (TRIPURA)"),
    "BELONIA": (23.25, 91.47, "BELONIA (TRIPURA)"),
    "BAGHJAN": (26.50, 93.90, "BAGHJAN OILFIELD (SIVASAGAR)"),
    "DIGBOI ROAD": (27.39, 95.62, "DIGBOI"),
    "HAYULIANG": (28.00, 96.06, "HAYULIANG (ANJAW)"),
    "HAWAI": (27.97, 96.31, "HAWAI (ANJAW)"),
    "JORAM": (27.96, 95.06, "JORAM (LOWER SIANG)"),
    "PANGIN": (28.15, 94.94, "PANGIN (SIANG)"),
    "BANDARDEWA": (27.11, 93.70, "BANDARDEWA (PAPUM PARE)"),
    "MAJULI": (26.96, 94.16, "MAJULI"),
    "BADARPUR": (24.87, 92.59, "BADARPUR (KARIMGANJ)"),
    "CHAKULIA": (22.50, 86.72, "CHAKULIA"),
    "PEREN": (25.46, 93.77, "PEREN (NAGALAND)"),
    "AKEGWO": (25.56, 94.48, "AKEGWO (MON)"),
    "AVANGKHU": (25.75, 94.40, "AVANGKHU (MON)"),
    "KHELLANI": (25.91, 94.26, "KHELLANI (MON)"),
    "KHANABAL": (25.93, 94.24, "KHANABAL (MON)"),
    "MON": (26.72, 95.03, "MON (NAGALAND)"),
    "NONGPOH": (25.90, 91.88, "NONGPOH (RI-BHOI)"),
    "UMIAM": (25.64, 91.89, "UMIAM (RI-BHOI)"),
    # -- education / institutions --
    "JAMMU": (32.7266, 74.8570, "JAMMU"),
    "VIJAYPUR": (32.60, 74.92, "VIJAYPUR (SAMBA)"),
    "SAMBA": (32.5637, 75.1000, "SAMBA"),
    "CUNCOLIM": (15.1700, 73.9080, "CUNCOLIM (SOUTH GOA)"),
    "NAHAN": (30.5596, 77.2939, "NAHAN (SIRMAUR)"),
    "SIRMAUR": (30.5596, 77.2939, "SIRMAUR"),
    "HYDERABAD INST": (17.38, 78.48, "HYDERABAD"),
    "PALANPUR": (24.1724, 72.4326, "PALANPUR"),
    "PAMPORE": (33.9965, 74.8835, "PAMPORE"),
    "PLADOR": (24.90, 92.60, "PLADOR (KARIMGANJ)"),
    # -- misc districts seen in data --
    "ALIPURDUAR": (26.4830, 89.5240, "ALIPURDUAR"),
    "COOCH BEHAR": (26.3455, 89.4450, "COOCH BEHAR"),
    "FAZILKA": (30.4039, 74.0280, "FAZILKA"),
    "DARBHANGA": (26.1542, 85.8918, "DARBHANGA"),
    "ERODE": (11.3410, 77.7172, "ERODE"),
    "UDHAGAMANDALAM": (11.4102, 76.6950, "UDHAGAMANDALAM"),
    "OOTY": (11.4102, 76.6950, "UDHAGAMANDALAM"),
    "NILGIRIS": (11.4102, 76.6950, "NILGIRIS"),
    "KANCHEEPURAM": (12.8352, 79.7046, "KANCHEEPURAM"),
    "CUDDALORE": (11.7480, 79.7641, "CUDDALORE"),
    "RAMANATHAPURAM": (9.3633, 78.8387, "RAMANATHAPURAM"),
    "KANYAKUMARI": (8.0883, 77.5385, "KANYAKUMARI"),
    "VILLUPURAM": (11.9400, 79.4930, "VILLUPURAM"),
    "SALEM": (11.6643, 78.1460, "SALEM"),
    "MADURAI": (9.9252, 78.1198, "MADURAI"),
    "TIRUNELVELI": (8.7139, 77.7567, "TIRUNELVELI"),
    "NAMAKKAL": (11.2189, 78.1671, "NAMAKKAL"),
    "KRISHNAGIRI": (12.5182, 78.2136, "KRISHNAGIRI"),
    "PANCHGRAM": (23.55, 90.05, "PANCHGRAM (FENI BORDER)"),
    "MAINAGURI": (26.56, 88.82, "MAINAGURI (JALPAIGURI)"),
    "JALPAIGURI": (26.5256, 88.7381, "JALPAIGURI"),
    "BHATINDA": (30.2110, 74.9379, "BATHINDA"),
    "RUPNAGAR": (30.9765, 76.5240, "RUPNAGAR"),
    "LUDHIANA": (30.9010, 75.8573, "LUDHIANA"),
    "PATHANKOT": (32.2646, 75.6437, "PATHANKOT"),
    "GURDASPUR": (32.0393, 75.4015, "GURDASPUR"),
    "MUKTSAR": (30.4742, 74.5141, "MUKTSAR"),
    "KAKINADA": (16.9891, 82.2475, "KAKINADA"),
    "NELLORE": (14.4426, 79.9865, "NELLORE"),
    "KURNOOL": (15.8281, 78.0373, "KURNOOL"),
    "ANANTAPUR": (14.6810, 77.6006, "ANANTAPUR"),
    "TIRUPATI": (13.6288, 79.4192, "TIRUPATI"),
    "GUNTUR": (16.3067, 80.4365, "GUNTUR"),
    "RAJAHMUNDRY": (17.0005, 81.8040, "RAJAHMUNDRY"),
    "ELURU": (16.7107, 81.0952, "ELURU"),
    "SRIKAKULAM": (18.2966, 83.8938, "SRIKAKULAM"),
    "KADAPA": (14.4674, 78.8242, "KADAPA"),
    "ONGOLE": (15.5057, 80.0499, "ONGOLE"),
    "CHITTOOR": (13.2172, 79.1003, "CHITTOOR"),
    "NALGONDA": (17.0678, 79.2671, "NALGONDA"),
    "WARANGAL": (17.9689, 79.5941, "WARANGAL"),
    "MAHABUBNAGAR": (16.7430, 78.0033, "MAHABUBNAGAR"),
    "KARIMNAGAR": (18.4386, 79.1288, "KARIMNAGAR"),
    "NIZAMABAD": (18.6745, 78.0982, "NIZAMABAD"),
    "ADILABAD": (19.6651, 78.5322, "ADILABAD"),
    "MEDAK": (17.9730, 78.3120, "MEDAK"),
    "SANGAREDDY": (17.6260, 78.0900, "SANGAREDDY"),
    # spring fewer states...
}

# --------------------------------------------------------------------------- #
# coal-company operating-area map (district-level fallback for mine names that
# carry no town token). Operating areas are public knowledge.
# --------------------------------------------------------------------------- #
COAL_ORG_DISTRICTS = {
    "BCCL": "DHANBAD",          # Bharat Coking Coal Ltd -> Dhanbad, Jharkhand
    "CCL":  "HAZARIBAGH",       # Central Coalfields -> Hazaribagh/Ramgarh belt
    "ECL":  "ASANSOL",          # Eastern Coalfields -> Raniganj (Asansol), WB
    "MCL":  "ANGUL",            # Mahanadi Coalfields -> Angul/Talcher, Odisha
    "NCL":  "SINGRAULI",        # Northern Coalfields -> Singrauli, MP
    "SECL": "BILASPUR",         # South Eastern Coalfields -> Bilaspur, CG
    "WCL":  "NAGPUR",           # Western Coalfields -> Nagpur belt, MH
    "SCCL": "KOTHAGUDEM",       # Singareni -> Kothagudem, Telangana
    "NMDC": "DANTEWADA",        # Bailadila iron ore -> Dantewada, CG
}

_LOC_LOOKUP = {re.sub(r"\s+", "", k): v for k, v in _DISTRICTS.items()}
_STOP_TOKENS = {
    "CHENNAI", "EXPRESSWAY", "HIGHWAY", "ROAD", "BYPASS", "CORRIDOR", "PKG",
    "PACKAGE", "PACKAGES", "SECTION", "SECTIONS", "LINE", "LINES", "DOUBLING",
    "TRACK", "TRACKS", "PORT", "AIRPORT", "TERMINAL", "QUAY", "JETTY", "GW",
    "NH", "NHIDCL", "NHAI", "AAI", "IIT", "IIM", "AIIMS", "IISER", "NIT",
    "PHASE", "PROJECT", "PROJECTS", "UNIT", "UNITS", "STAGE", "LATITUDE",
    "BRIDGE", "TUNNEL", "REHABILITATION", "UPGRADATION", "EXTENSION",
}


def geocode_project(project_name) -> tuple[Optional[float], Optional[float], str]:
    """District/town-level geocode of a project name.

    Strategy (best match wins):
      1. longest matching table token run inside the name (whitespace-normalised);
      2. coal-company operating-area fallback for mining orgs in the name;
      3. (None, None, "") if nothing matched -> caller keeps state centroid.
    Returns (lat, lon, matched_token).
    """
    if project_name is None or (isinstance(project_name, float) and math.isnan(project_name)):
        return None, None, ""
    name = str(project_name)
    nkey = re.sub(r"[^A-Z0-9]", "", name.upper())

    # 1. longest town-table hit (try full current-char string then sliding windows
    #    is overkill; we do a greedy longest-substring over the token run)
    tokens = re.findall(r"[A-Z][A-Z0-9\-]{2,}", name)
    n = len(tokens)
    best = (0, None)
    for start in range(n):
        for k in range(min(n - start, 6), 0, -1):
            seq = "".join(re.sub(r"[^A-Z0-9]", "", t) for t in tokens[start:start + k])
            hit = _LOC_LOOKUP.get(seq)
            if hit is None and k == 1:
                pass
            elif hit is not None:
                if k > best[0]:
                    best = (k, hit)
                break
    if best[1] is not None:
        lat, lon, disp = best[1]
        return float(lat), float(lon), disp

    # 2. coal-org fallback for mine/site rows with no town token
    upper = name.upper()
    for org, dist in COAL_ORG_DISTRICTS.items():
        if org in upper:
            lat, lon, disp = _LOC_LOOKUP[dist]
            return float(lat), float(lon), disp
    return None, None, ""


def enrich_generator():
    """Standalone (lat, lon, token) for the CSV backfill without pandas deps."""
    return geocode_project