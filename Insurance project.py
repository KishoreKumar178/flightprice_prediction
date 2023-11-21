import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.title("Insurance Project")
st.subheader("Buying Prediction 1")
st.write("Check whether you will take travel insurance or not")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=0)
    Employment_type = st.selectbox("Employment type",['Government Sector','Private Sector/Self Employed'])
    Graduate_status = st.selectbox("Graduate", ['Yes','No'])
    chronic_disease = st.selectbox("chronic disease", ['Yes','No'])
    
with col2:
    Annual_income = st.number_input("Annual income", min_value= 0)
    Family_members = st.number_input("Family members", min_value= 1)
    frequent_flyer = st.selectbox("frequent flyer", ['Yes','No'])
    EverTravelledAbroad = st.selectbox("EverTravelledAbroad", ['Yes','No'])

with open (r"C:\Users\Kishore kumar\Downloads\Python program\model.pkl","rb") as f:
    model = pickle.load(f)
with open (r"C:\Users\Kishore kumar\Downloads\Python program\scaler1.pkl","rb") as f:
    scaler = pickle.load(f)

# preprocessing
age = int(age)
d = {'Age':[age],'Employment_type': [Employment_type], 'Graduate_status':[Graduate_status], 
                    'Annual_income':[Annual_income],'Family_members':[Family_members],
                    "chronic_disease":[chronic_disease],
                    'frequent_flyer':[frequent_flyer],'EverTravelledAbroad':[EverTravelledAbroad]}
data = pd.DataFrame(data = d, index = [0])
data['Age'] = data['Age'].astype('int')
data['Annual_income'] = data['Annual_income'].astype('int')
data['Employment_type'] = data['Employment_type'].map({'Government Sector':0,'Private Sector/Self Employed':1})
data['Graduate_status'] = data['Graduate_status'].map({'Yes':1,'No':0})
data['chronic_disease'] = data['chronic_disease'].map({'Yes':1,'No':0})
data['frequent_flyer'] = data['frequent_flyer'].map({'Yes':1,'No':0})
data['EverTravelledAbroad'] = data['EverTravelledAbroad'].map({'Yes':1,'No':0})

data = scaler.transform(data)
if st.button("Predict"):
    result = model.predict(data)
    if result == 0:
        st.write("You will not take travel insurance")
    else:
        st.success("You will take travel insurance")

st.subheader("Buying Prediction 2")
st.write("Check whether you will take travel insurance or not")
col1, col2 = st.columns(2)
with col1:
    PAXCOUNT = st.number_input("PAXCOUNT", min_value=0, max_value=9)
    Sales_channel = st.selectbox("Sales channel",['Internet','Mobile'])
    Trip_type = st.selectbox("Trip_TYPE", ['RoundTrip','CircleTrip','OneWay'])
    Purchase_Lead = st.number_input("Purchase Lead", min_value= 0)
    lenght_of_stay = st.number_input("lenght of stay",min_value= 0)
    FNB = st.selectbox("FNB", ["Yes","No"])
    flight_duration = st.selectbox("Duration",[5.52, 5.07, 7.57, 6.62, 7,  4.75, 8.83, 7.42, 6.42, 5.33, 4.67, 5.62, 8.58, 8.67,
                                               4.72, 8.15, 6.33, 5, 4.83, 9.5, 5.13])

with col2:
    Flight_hour = st.number_input("Flight hour", min_value=0, max_value=24)
    Flight_day = st.selectbox("Flight_day", ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
    Route = st.selectbox("Route", ['AKLDEL','AKLHGH','AKLHND','AKLICN','AKLKIX','AKLKTM','AKLKUL','AKLMRU',
'AKLPEK','AKLPVG','AKLTPE','AORICN','AORKIX','AORKTM','AORMEL','BBIMEL',
'BBIOOL','BBIPER','BBISYD','BDOCTS','BDOCTU','BDOHGH','BDOICN','BDOIKA',
'BDOKIX','BDOMEL','BDOOOL','BDOPEK','BDOPER','BDOPUS','BDOPVG','BDOSYD',
'BDOTPE','BDOXIY','BKICKG','BKICTS','BKICTU','BKIHND','BKIICN','BKIKIX',
'BKIKTM','BKIMEL','BKIMRU','BKIOOL','BKIPEK','BKIPER','BKIPUS','BKIPVG',
'BKISYD','BKIXIY','BLRICN','BLRMEL','BLRPER','BLRSYD','BOMMEL','BOMOOL',
'BOMPER','BOMSYD','BTJJED','BTUICN','BTUPER','BTUSYD','BTUWUH','BWNCKG',
'BWNDEL','BWNHGH','BWNIKA','BWNKTM','BWNMEL','BWNOOL','BWNPER','BWNSYD',
'BWNTPE','CANDEL','CANIKA','CANMEL','CANMRU','CANOOL','CANPER','CANSYD',
'CCUMEL','CCUMRU','CCUOOL','CCUPER','CCUSYD','CCUTPE','CEBMEL','CEBOOL',
'CEBPER','CEBSYD','CGKCKG','CGKCTS','CGKCTU','CGKDEL','CGKHGH','CGKHND',
'CGKICN','CGKIKA','CGKJED','CGKKIX','CGKKTM','CGKMEL','CGKMRU','CGKOOL',
'CGKPEK','CGKPER','CGKPUS','CGKPVG','CGKSYD','CGKTPE','CGKWUH','CGKXIY',
'CKGCOK','CKGDPS','CKGJHB','CKGKCH','CKGLOP','CKGMAA','CKGMEL','CKGMYY',
'CKGOOL','CKGPEN','CKGPER','CKGPNH','CKGSBW','CKGSIN','CKGSUB','CKGSYD',
'CKGTGG','CKGTRZ','CKGTWU','CMBCTS','CMBCTU','CMBHGH','CMBHND','CMBICN',
'CMBKIX','CMBMEL','CMBMRU','CMBOOL','CMBPEK','CMBPER','CMBPVG','CMBSYD',
'CMBWUH','CNXHND','CNXICN','CNXKIX','CNXMEL','CNXOOL','CNXPEK','CNXPER',
'CNXPVG','CNXSYD','CNXTPE','COKCTU','COKHGH','COKICN','COKKIX','COKMEL',
'COKOOL','COKPER','COKPUS','COKSYD','COKTPE','COKWUH','CRKMEL','CRKOOL',
'CRKSYD','CSXPER','CTSDMK','CTSDPS','CTSHKT','CTSJHB','CTSKBR','CTSKCH',
'CTSKNO','CTSLGK','CTSMEL','CTSMYY','CTSOOL','CTSPEN','CTSPER','CTSSGN',
'CTSSIN','CTSSUB','CTSSYD','CTUDPS','CTUHKT','CTUIKA','CTUJHB','CTUKBV',
'CTUKCH','CTUKNO','CTUMAA','CTUMEL','CTUMRU','CTUMYY','CTUOOL','CTUPEN',
'CTUPER','CTUSBW','CTUSIN','CTUSUB','CTUSYD','CTUTGG','CTUTRZ','CTUTWU',
'CXRMEL','DACHGH','DACHND','DACICN','DACKIX','DACMEL','DACOOL','DACPER',
'DACSYD','DACTPE','DADMEL','DADOOL','DADSYD','DELDMK','DELDPS','DELHKG',
'DELHKT','DELHND','DELJHB','DELJOG','DELKBV','DELKCH','DELKIX','DELKNO',
'DELLGK','DELMEL','DELMFM','DELMNL','DELMRU','DELMYY','DELOOL','DELPEN',
'DELPER','DELPNH','DELSBW','DELSGN','DELSIN','DELSUB','DELSYD','DELSZX',
'DMKHGH','DMKHND','DMKICN','DMKIKA','DMKKIX','DMKKTM','DMKMEL','DMKMRU',
'DMKOOL','DMKPEK','DMKPER','DMKPUS','DMKPVG','DMKSYD','DMKTPE','DPSHGH',
'DPSHND','DPSICN','DPSIKA','DPSKIX','DPSKTM','DPSMEL','DPSMRU','DPSOOL',
'DPSPEK','DPSPUS','DPSPVG','DPSSYD','DPSTPE','DPSXIY','GOIKUL','GOIMEL',
'GOIOOL','GOIPER','GOISYD','HANKTM','HANMEL','HANOOL','HANPER','HANSYD',
'HDYHGH','HDYKTM','HDYMEL','HDYOOL','HDYPEK','HDYPER','HDYPVG','HDYSYD',
'HDYTPE','HGHHKT','HGHJHB','HGHJOG','HGHKBR','HGHKBV','HGHKCH','HGHKNO',
'HGHLGK','HGHLOP','HGHMAA','HGHMEL','HGHMYY','HGHOOL','HGHPEN','HGHPER',
'HGHSBW','HGHSUB','HGHSYD','HGHTRZ','HKGIKA','HKGKTM','HKGMEL','HKGMRU',
'HKGOOL','HKGPER','HKGSYD','HKTHND','HKTICN','HKTKIX','HKTKTM','HKTMEL',
'HKTMRU','HKTOOL','HKTPEK','HKTPER','HKTPUS','HKTPVG','HKTSYD','HKTTPE',
'HKTXIY','HNDIKA','HNDJOG','HNDKBR','HNDKBV','HNDKCH','HNDKNO','HNDKTM',
'HNDLGK','HNDLOP','HNDMAA','HNDMEL','HNDMLE','HNDOOL','HNDPEN','HNDPER',
'HNDPNH','HNDREP','HNDRGN','HNDSBW','HNDSGN','HNDSIN','HNDSUB','HNDSYD',
'HNDTRZ','HYDMEL','HYDOOL','HYDPER','HYDSYD','HYDWUH','ICNIKA','ICNJED',
'ICNJHB','ICNKBR','ICNKBV','ICNKCH','ICNKNO','ICNKTM','ICNLGK','ICNMAA',
'ICNMEL','ICNMLE','ICNMYY','ICNOOL','ICNPEN','ICNPER','ICNREP','ICNRGN',
'ICNSBW','ICNSDK','ICNSGN','ICNSIN','ICNSUB','ICNSYD','ICNTRZ','ICNVTZ',
'IKAKCH','IKAKIX','IKALOP','IKAMEL','IKAMFM','IKAMNL','IKAOOL','IKAPEK',
'IKAPEN','IKAPER','IKAPUS','IKAPVG','IKASGN','IKASIN','IKASUB','IKASYD',
'IKATPE','JEDJOG','JEDKNO','JEDMEL','JEDMNL','JEDPDG','JEDPEN','JEDSUB',
'JHBKIX','JHBKTM','JHBMEL','JHBMRU','JHBPEK','JHBPUS','JHBPVG','JHBSYD',
'JHBTPE','JHBWUH','JHBXIY','JOGKIX','JOGKTM','JOGMEL','JOGOOL','JOGPER',
'JOGPVG','JOGSYD','JOGTPE','KBRKIX','KBRKTM','KBRMEL','KBROOL','KBRPEK',
'KBRPER','KBRPVG','KBRSYD','KBRTPE','KBVKTM','KBVMEL','KBVOOL','KBVPEK',
'KBVPER','KBVPVG','KBVSYD','KCHKIX','KCHKTM','KCHMEL','KCHMRU','KCHOOL',
'KCHPEK','KCHPER','KCHPUS','KCHPVG','KCHSYD','KCHTPE','KCHXIY','KHHMEL',
'KHHOOL','KHHPER','KHHSYD','KIXKNO','KIXKTM','KIXLGK','KIXLOP','KIXMAA',
'KIXMEL','KIXMLE','KIXMYY','KIXOOL','KIXPEN','KIXPER','KIXPNH','KIXREP',
'KIXRGN','KIXSBW','KIXSGN','KIXSIN','KIXSUB','KIXSYD','KIXTGG','KIXTRZ',
'KLOMEL','KLOOOL','KNOKTM','KNOMEL','KNOOOL','KNOPEK','KNOPER','KNOPUS',
'KNOPVG','KNOSYD','KNOTPE','KNOXIY','KOSMEL','KOSOOL','KOSPEK','KOSSYD',
'KTMMEL','KTMMFM','KTMMYY','KTMPEN','KTMPER','KTMREP','KTMSGN','KTMSIN',
'KTMSUB','KTMSYD','KTMTGG','KTMTPE','KTMURT','KWLPER','LBUPER','LGKMEL',
'LGKOOL','LGKPER','LGKPUS','LGKPVG','LGKSYD','LGKTPE','LOPOOL','LOPPEK',
'LOPPVG','LOPSYD','LOPTPE','LOPXIY','LPQMEL','LPQOOL','LPQPER','LPQTPE',
'MAAMEL','MAAMRU','MAAOOL','MAAPER','MAAPVG','MAASYD','MAATPE','MAAWUH',
'MELMFM','MELMLE','MELMNL','MELMRU','MELMYY','MELPEK','MELPEN','MELPNH',
'MELPUS','MELPVG','MELREP','MELRGN','MELSBW','MELSGN','MELSIN','MELSUB',
'MELSWA','MELSZX','MELTGG','MELTPE','MELTRZ','MELTWU','MELURT','MELUTP',
'MELVTE','MELVTZ','MELWUH','MELXIY','MFMOOL','MFMPER','MFMSYD','MLEPEK',
'MLEPER','MLESYD','MNLMRU','MNLOOL','MNLPER','MNLSYD','MRUOOL','MRUPEK',
'MRUPEN','MRUPER','MRUPVG','MRUSGN','MRUSIN','MRUSUB','MRUSYD','MRUSZX',
'MYYOOL','MYYPER','MYYPUS','MYYSYD','MYYXIY','NRTSYD','OOLPEK','OOLPEN',
'OOLPNH','OOLPUS','OOLPVG','OOLREP','OOLRGN','OOLSBW','OOLSDK','OOLSGN',
'OOLSIN','OOLSUB','OOLSZX','OOLTGG','OOLTPE','OOLTRZ','OOLTWU','OOLURT',
'OOLUTP','OOLVTE','OOLWUH','OOLXIY','PEKPEN','PEKPER','PEKREP','PEKRGN',
'PEKSBW','PEKSIN','PEKSUB','PEKSYD','PEKTGG','PEKTRZ','PEKTWU','PENPER',
'PENPUS','PENPVG','PENSYD','PENTPE','PENWUH','PENXIY','PERPNH','PERPUS',
'PERPVG','PERREP','PERRGN','PERSBW','PERSDK','PERSGN','PERSIN','PERSWA',
'PERSZX','PERTGG','PERTPE','PERTRZ','PERTWU','PERUTP','PERVTE','PERVTZ',
'PERWUH','PERXIY','PNHSYD','PNHTPE','PNKTPE','PUSRGN','PUSSBW','PUSSGN',
'PUSSIN','PUSSUB','PUSSYD','PUSTRZ','PVGREP','PVGRGN','PVGSIN','PVGSUB',
'PVGSYD','PVGTGG','PVGTWU','PVGURT','REPSYD','REPTPE','RGNSYD','RGNTPE',
'SBWSYD','SBWTPE','SBWXIY','SDKSYD','SGNSYD','SGNXIY','SINSYD','SINTPE',
'SINWUH','SINXIY','SRGTPE','SUBSYD','SUBTPE','SUBXIY','SYDSZX','SYDTPE',
'SYDTRZ','SYDTWU','SYDVTE','SYDVTZ','SYDXIY','TGGTPE','TGGXIY','TPETRZ',
'TPEVTE','TRZWUH','TRZXIY','TWUXIY','HGHSGN','ICNTGG','JHBOOL','KBRXIY',
'KBVTPE','KIXTWU','LBUTPE','PVGSGN','SBWWUH','DELREP','DPSWUH','HKGJED',
'KBVKIX','KBVPUS','KIXLPQ','LGKPEK','LGKXIY','LOPPER','PEKSGN','PERSUB',
'TPETWU','BDOWUH','BKIDEL','CKGSGN','CTUKBR','CTULGK','CTUREP','DACMRU',
'DACPEK','DELRGN','HDYXIY','HGHTGG','HKTWUH','ICNVTE','KBRPUS','KCHWUH',
'KLOSYD','KNOWUH','MLETPE','SDKTPE','SUBWUH','TWUWUH','AORPUS','BTUCKG',
'BWNWUH','CKGKNO','CKGLGK','CNXDEL','CNXPUS','CTSJOG','CTSSBW','CTUDMK',
'CTULOP','DELKBR','DELURT','HDYKIX','HGHSIN','HGHTWU','HYDMRU','IKASZX',
'KBVWUH','KBVXIY','KIXLBU','LGKWUH','MELNRT','MLEOOL','MRUTPE','TPEURT',
'URTXIY','AORPER','CKGHKT','CKGMRU','CNXXIY','COKCTS','CSXMRU','CSXSYD',
'CTUMLE','CTUSGN','CTUSRG','CTUURT','DACPUS','HGHMRU','HKTIKA','HKTJED',
'ICNMRU','JEDMFM','KBRWUH','KIXMRU','KTMTWU','MLEPVG','MRUXIY'])
    
    Country = st.selectbox("Country", ['New Zealand','India','United Kingdom','China','South Korea','Japan',
'Malaysia','Singapore','Switzerland','Germany','Indonesia',
'Czech Republic','Vietnam','Thailand','Spain','Romania','Ireland','Italy',
'Slovakia','United Arab Emirates','Tonga','Réunion','(not set)',
'Saudi Arabia','Netherlands','Qatar','Hong Kong','Philippines',
'Sri Lanka','France','Croatia','United States','Laos','Hungary',
'Portugal','Cyprus','Australia','Cambodia','Poland','Belgium','Oman',
'Bangladesh','Kazakhstan','Brazil','Turkey','Kenya','Taiwan','Brunei',
'Chile','Bulgaria','Ukraine','Denmark','Colombia','Iran','Bahrain',
'Solomon Islands','Slovenia','Mauritius','Nepal','Russia','Kuwait',
'Mexico','Sweden','Austria','Lebanon','Jordan','Greece','Mongolia',
'Canada','Tanzania','Peru','Timor-Leste','Argentina','New Caledonia',
'Macau','Myanmar (Burma)','Norway','Panama','Bhutan','Norfolk Island',
'Finland','Nicaragua','Maldives','Egypt','Israel','Tunisia',
'South Africa','Papua New Guinea','Paraguay','Estonia','Seychelles',
'Afghanistan','Guam','Czechia','Malta','Vanuatu','Belarus','Pakistan',
'Iraq','Ghana','Gibraltar','Guatemala','Algeria','Svalbard & Jan Mayen'])
    baggage = st.selectbox("Baggage", ["Yes","No"])
    seat = st.selectbox("Seat", ["Yes","No"])

# preprocessing
PAXCOUNT = int(PAXCOUNT)
Flight_hour = int(Flight_hour)
Purchase_Lead = int(Purchase_Lead)
lenght_of_stay = int(lenght_of_stay)
d1 = {'PAXCOUNT':[PAXCOUNT],'SALESCHANNEL': [Sales_channel], 'Trip_type':[Trip_type], 
                    'Purchase_Lead':[Purchase_Lead],'lenght_of_stay':[lenght_of_stay],
                    "Flight_hour":[Flight_hour],"Flight_day":[Flight_day],"ROUTE":[Route],
                    "geonetwork_country":[Country],"baggage":[baggage],"seat":[seat],'FNB':[FNB],'flight_duration':[flight_duration]
                    }
data = pd.DataFrame(data = d1, index = [0])
data['Flight_day'] = data['Flight_day'].map({'Mon':1,'Tue':2,'Wed':3,'Thu':4,'Fri':5,'Sat':6,'Sun':7})
data['PAXCOUNT'] = data['PAXCOUNT'].astype('int')
data['Flight_hour'] = data['Flight_hour'].astype('int')
data['Purchase_Lead'] = data['Purchase_Lead'].astype('int')
data['lenght_of_stay'] = data['lenght_of_stay'].astype('int')
data['SALESCHANNEL'] = data['SALESCHANNEL'].map({'Internet':1,'Mobile':0})
data['Trip_type'] = data['Trip_type'].astype('category')
data['Flight_day'] = data['Flight_day'].astype('category')
data["ROUTE"] = data["ROUTE"].astype('category')
data["geonetwork_country"] = data["geonetwork_country"].astype('category')
data['baggage'] = data['baggage'].map({'Yes':1,'No':0})
data['seat'] = data['seat'].map({'Yes':1,'No':0})
data['FNB'] = data['FNB'].map({'Yes':1,'No':0})


with open (r"C:\Users\Kishore kumar\Downloads\Python program\Travel insurance.pkl","rb") as f:
    model1 = pickle.load(f)
with open (r"C:\Users\Kishore kumar\Downloads\Python program\ohe1.pkl","rb") as f:
    ohe = pickle.load(f)
with open (r"C:\Users\Kishore kumar\Downloads\Python program\scaler2.pkl","rb") as f:
    scaler1 = pickle.load(f)

d1 = ohe.transform(data[["Flight_day","ROUTE","geonetwork_country","Trip_type"]]).toarray()
d1 = pd.DataFrame(d1)
data.drop(["Flight_day","ROUTE","geonetwork_country","Trip_type"],axis=1,inplace=True)
data = pd.concat([data,d1],axis=1)
d1 = scaler1.transform(data)
result = model1.predict(d1)
if st.button("Predict1"):
    result = model1.predict(d1)
    if result == 0:
        st.write("You will not take travel insurance")
    else:
        st.success("You will take travel insurance")

st.subheader("Claim Prediction")
st.write("Check whether the passenger will claim or not")
col1, col2 = st.columns(2)
with col1:
    Agency_type = st.selectbox("Agency_type", ['Travel Agency','Airlines'])             
    Agency = st.selectbox("Agency_Type", ['CBH', 'CWT', 'JZI', 'KML', 'EPX', 'C2B', 'JWT', 'RAB', 'SSI',
       'ART', 'CSR', 'CCR', 'ADM', 'LWC', 'TTW', 'TST'])               
    Commision = st.number_input("Commision")              
    Age1 = st.number_input("Age", min_value= 0, key = 6)
    Net_Sales = st.number_input("Net_Sales") 
with col2:
    Distribution_Channel= st.selectbox("Distribution_Channel", ['Online','Offline'])
    Product_Name = st.selectbox("Product_Name", ['Comprehensive Plan', 'Rental Vehicle Excess Insurance',
       'Value Plan', 'Basic Plan', 'Premier Plan',
       '2 way Comprehensive Plan', 'Bronze Plan', 'Silver Plan',
       'Annual Silver Plan', 'Cancellation Plan',
       '1 way Comprehensive Plan', 'Ticket Protector', '24 Protect',
       'Gold Plan', 'Annual Gold Plan',
       'Single Trip Travel Protect Silver',
       'Individual Comprehensive Plan',
       'Spouse or Parents Comprehensive Plan',
       'Annual Travel Protect Silver',
       'Single Trip Travel Protect Platinum',
       'Annual Travel Protect Gold', 'Single Trip Travel Protect Gold',
       'Annual Travel Protect Platinum', 'Child Comprehensive Plan',
       'Travel Cruise Protect', 'Travel Cruise Protect Family']) 
    Duration = st.number_input("Duration", min_value= 0)         
    Destination = st.selectbox("Destination", ['MALAYSIA', 'AUSTRALIA', 'ITALY', 'UNITED STATES', 'THAILAND',
       "KOREA, DEMOCRATIC PEOPLE'S REPUBLIC OF", 'NORWAY', 'VIET NAM',
       'DENMARK', 'SINGAPORE', 'JAPAN', 'UNITED KINGDOM', 'INDONESIA',
       'INDIA', 'CHINA', 'FRANCE', 'TAIWAN, PROVINCE OF CHINA',
       'PHILIPPINES', 'MYANMAR', 'HONG KONG', 'KOREA, REPUBLIC OF',
       'UNITED ARAB EMIRATES', 'NAMIBIA', 'NEW ZEALAND', 'COSTA RICA',
       'BRUNEI DARUSSALAM', 'POLAND', 'SPAIN', 'CZECH REPUBLIC',
       'GERMANY', 'SRI LANKA', 'CAMBODIA', 'AUSTRIA', 'SOUTH AFRICA',
       'TANZANIA, UNITED REPUBLIC OF', "LAO PEOPLE'S DEMOCRATIC REPUBLIC",
       'NEPAL', 'NETHERLANDS', 'MACAO', 'CROATIA', 'FINLAND', 'CANADA',
       'TUNISIA', 'RUSSIAN FEDERATION', 'GREECE', 'BELGIUM', 'IRELAND',
       'SWITZERLAND', 'CHILE', 'ISRAEL', 'BANGLADESH', 'ICELAND',
       'PORTUGAL', 'ROMANIA', 'KENYA', 'GEORGIA', 'TURKEY', 'SWEDEN',
       'MALDIVES', 'ESTONIA', 'SAUDI ARABIA', 'PAKISTAN', 'QATAR', 'PERU',
       'LUXEMBOURG', 'MONGOLIA', 'ARGENTINA', 'CYPRUS', 'FIJI',
       'BARBADOS', 'TRINIDAD AND TOBAGO', 'ETHIOPIA', 'PAPUA NEW GUINEA',
       'SERBIA', 'JORDAN', 'ECUADOR', 'BENIN', 'OMAN', 'BAHRAIN',
       'UGANDA', 'BRAZIL', 'MEXICO', 'HUNGARY', 'AZERBAIJAN', 'MOROCCO',
       'URUGUAY', 'MAURITIUS', 'JAMAICA', 'KAZAKHSTAN', 'GHANA',
       'UZBEKISTAN', 'SLOVENIA', 'KUWAIT', 'GUAM', 'BULGARIA',
       'LITHUANIA', 'NEW CALEDONIA', 'EGYPT', 'ARMENIA', 'BOLIVIA',
       'VIRGIN ISLANDS, U.S.', 'PANAMA', 'SIERRA LEONE', 'COLOMBIA',
       'PUERTO RICO', 'UKRAINE', 'GUINEA', 'GUADELOUPE',
       'MOLDOVA, REPUBLIC OF', 'GUYANA', 'LATVIA', 'ZIMBABWE', 'VANUATU',
       'VENEZUELA', 'BOTSWANA', 'BERMUDA', 'MALI', 'KYRGYZSTAN',
       'CAYMAN ISLANDS', 'MALTA', 'LEBANON', 'REUNION', 'SEYCHELLES',
       'ZAMBIA', 'SAMOA', 'NORTHERN MARIANA ISLANDS', 'NIGERIA',
       'DOMINICAN REPUBLIC', 'TAJIKISTAN', 'ALBANIA',
       'MACEDONIA, THE FORMER YUGOSLAV REPUBLIC OF',
       'LIBYAN ARAB JAMAHIRIYA', 'ANGOLA', 'BELARUS',
       'TURKS AND CAICOS ISLANDS', 'FAROE ISLANDS', 'TURKMENISTAN',
       'GUINEA-BISSAU', 'CAMEROON', 'BHUTAN', 'RWANDA', 'SOLOMON ISLANDS',
       'IRAN, ISLAMIC REPUBLIC OF', 'GUATEMALA', 'FRENCH POLYNESIA',
       'TIBET', 'SENEGAL', 'REPUBLIC OF MONTENEGRO',
       'BOSNIA AND HERZEGOVINA'])  
        
# preprocessing
data2 = {'Agency Type':[Agency_type], "Distribution Channel":[Distribution_Channel],"Duration":[Duration],"Net Sales":[Net_Sales],
         "Commision":[Commision],"Age":[Age1],"Product Name":[Product_Name],"Destination":[Destination], "Agency":[Agency]}
df2 = pd.DataFrame(data2)
df2['Age'] = df2['Age'].astype('int')
df2['Net Sales'] = df2['Net Sales'].astype('float')
df2['Commision'] = df2['Commision'].astype('float')
df2['Duration'] = df2['Duration'].astype('int')
df2['Agency Type'] = df2['Agency Type'].map({'Travel Agency':1,'Airlines':0})
df2['Distribution Channel'] = df2['Distribution Channel'].map({'Online':1,'Offline':0})
df2['Product Name'] = df2['Product Name'].astype('category')
df2['Destination'] = df2['Destination'].astype('category')
df2['Agency'] = df2['Agency'].astype('category')

with open (r"C:\Users\Kishore kumar\Downloads\Python program\ohe3.pkl","rb") as f:
    ohe3 = pickle.load(f)
with open (r"C:\Users\Kishore kumar\Downloads\Python program\scaler3.pkl","rb") as f:
    scaler3 = pickle.load(f)
with open (r"C:\Users\Kishore kumar\Downloads\Python program\Claim.pkl","rb") as f:
    model3 = pickle.load(f)
d1 = ohe3.transform(df2[["Agency","Product Name","Destination"]]).toarray()
d1 = pd.DataFrame(d1)
df2.drop(["Agency","Product Name","Destination"],axis=1,inplace=True)
df2 = pd.concat([df2,d1],axis=1)
d1 = scaler3.transform(df2)
if st.button("Predict 3"):
    result = model3.predict(d1)
    if result == 0:
        st.write("You will not claim")
    else:
        st.success("You will claim")