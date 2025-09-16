import streamlit as st

TRANSLATIONS = {
    "en": {
        "app_title": "Smart Agriculture AI Dashboard",
        "app_tagline": "A one-stop AI assistant for crop planning, plant health, hyperspectral analysis, and weather insights.",
        "step1_header": "📍 Step 1: Enter Your Farm Details",
        "step1_info": "Your location is used to automatically fetch weather data and provide AI-powered suggestions.",
        "step2_header": "🌾 Step 2: AI Crop & Climate Report",
        "step2_info": "Get a unified report with detailed climate analysis and crop suggestions for your farm.",
        "step3_header": "🍃 Step 3: Plant Health Check",
        "step3_info": "Upload a leaf photo to check for plant diseases and get treatment suggestions.",
        "step4_header": "🌈 Step 4: Hyperspectral Data Analysis",
        "step4_info": "Paste hyperspectral reflectance data to classify crops or land cover using a trained ONNX model.",
        "step5_header": "☀️ Step 5: Weather Insights",
        "step5_info": "Get the latest 30-day weather data from NASA POWER for your farm's location.",
        "step6_header": "📷 Step 6: Live Disease Detection",
        "step6_info": "Use your camera to detect plant diseases in real-time.",
        "step7_header": "📈 Step 7: Historical Weather Analysis",
        "step7_info": "Analyze the last 5 years of weather data for your location to understand long-term trends.",
    },
    "hi": {
        "app_title": "स्मार्ट कृषि एआई डैशबोर्ड",
        "app_tagline": "फसल योजना, पौधे के स्वास्थ्य, हाइपरस्पेक्ट्रल विश्लेषण और मौसम की जानकारी के लिए एक-स्टॉप एआई सहायक।",
        "step1_header": "📍 चरण 1: अपने खेत का विवरण दर्ज करें",
        "step1_info": "आपके स्थान का उपयोग स्वचालित रूप से मौसम डेटा प्राप्त करने और एआई-संचालित सुझाव प्रदान करने के लिए किया जाता है।",
        "step2_header": "🌾 चरण 2: एआई फसल और जलवायु रिपोर्ट",
        "step2_info": "अपने खेत के लिए विस्तृत जलवायु विश्लेषण और फसल सुझावों के साथ एक एकीकृत रिपोर्ट प्राप्त करें।",
        "step3_header": "🍃 चरण 3: पौधे के स्वास्थ्य की जाँच करें",
        "step3_info": "पौधों की बीमारियों की जांच करने और उपचार के सुझाव प्राप्त करने के लिए एक पत्ते की तस्वीर अपलोड करें।",
        "step4_header": "🌈 चरण 4: हाइपरस्पेक्ट्रल डेटा विश्लेषण",
        "step4_info": "प्रशिक्षित ONNX मॉडल का उपयोग करके फसलों या भूमि कवर को वर्गीकृत करने के लिए हाइपरस्पेक्ट्रल परावर्तन डेटा पेस्ट करें।",
        "step5_header": "☀️ चरण 5: मौसम की जानकारी",
        "step5_info": "अपने खेत के स्थान के लिए नासा पावर से नवीनतम 30-दिन का मौसम डेटा प्राप्त करें।",
        "step6_header": "📷 चरण 6: लाइव रोग का पता लगाना",
        "step6_info": "वास्तविक समय में पौधों की बीमारियों का पता लगाने के लिए अपने कैमरे का उपयोग करें।",
        "step7_header": "📈 चरण 7: ऐतिहासिक मौसम विश्लेषण",
        "step7_info": "दीर्घकालिक रुझानों को समझने के लिए अपने स्थान के लिए पिछले 5 वर्षों के मौसम डेटा का विश्लेषण करें।",
    },
    "mr": {
        "app_title": "स्मार्ट शेती एआय डॅशबोर्ड",
        "app_tagline": "पीक नियोजन, वनस्पती आरोग्य, हायपरस्पेक्ट्रल विश्लेषण आणि हवामान अंतर्दृष्टीसाठी एक-स्टॉप एआय सहाय्यक.",
        "step1_header": "📍 पायरी 1: तुमच्या शेतीचा तपशील प्रविष्ट करा",
        "step1_info": "तुमचे स्थान स्वयंचलितपणे हवामान डेटा आणण्यासाठी आणि AI-शक्तीशाली सूचना देण्यासाठी वापरले जाते.",
        "step2_header": "🌾 पायरी 2: AI पीक आणि हवामान अहवाल",
        "step2_info": "तुमच्या शेतासाठी तपशीलवार हवामान विश्लेषण आणि पीक सूचनांसह एकत्रित अहवाल मिळवा.",
        "step3_header": "🍃 पायरी 3: वनस्पती आरोग्य तपासा",
        "step3_info": "वनस्पतींच्या रोगांची तपासणी करण्यासाठी आणि उपचार सूचना मिळवण्यासाठी पानाचा फोटो अपलोड करा.",
        "step4_header": "🌈 पायरी 4: हायपरस्पेक्ट्रल डेटा विश्लेषण",
        "step4_info": "प्रशिक्षित ONNX मॉडेल वापरून पिके किंवा जमीन आच्छादन वर्गीकृत करण्यासाठी हायपरस्पेक्ट्रल परावर्तन डेटा पेस्ट करा.",
        "step5_header": "☀️ पायरी 5: हवामान अंतर्दृष्टी",
        "step5_info": "तुमच्या शेताच्या स्थानासाठी नासा पॉवरकडून नवीनतम 30-दिवसांचा हवामान डेटा मिळवा.",
        "step6_header": "📷 पायरी 6: थेट रोग ओळख",
        "step6_info": "रिअल-टाइममध्ये वनस्पतींचे रोग शोधण्यासाठी तुमचा कॅमेरा वापरा.",
        "step7_header": "📈 पायरी 7: ऐतिहासिक हवामान विश्लेषण",
        "step7_info": "दीर्घकालीन ट्रेंड समजून घेण्यासाठी तुमच्या स्थानासाठी गेल्या 5 वर्षांच्या हवामान डेटाचे विश्लेषण करा.",
    }
}

def t(key):
    """
    Returns the translated string for the given key in the selected language.
    """
    if 'lang' not in st.session_state:
        st.session_state['lang'] = 'en'

    return TRANSLATIONS[st.session_state['lang']].get(key, key)
