FERTILIZER_RECOMMENDATIONS = {
    "Alluvial Soil": {
        "Wheat": "Recommended NPK ratio: 120:60:40. Use Urea, DAP, and MOP.",
        "Paddy": "Recommended NPK ratio: 120:60:60. Use Urea, SSP, and MOP.",
        "Sugarcane": "Recommended NPK ratio: 250:80:80. Use Urea, SSP, and MOP.",
        "Cotton": "Recommended NPK ratio: 150:60:60. Use Urea, DAP, and MOP.",
    },
    "Black (Regur) Soil": {
        "Cotton": "Recommended NPK ratio: 120:60:60. Use Urea, DAP, and MOP.",
        "Soybean": "Recommended NPK ratio: 20:60:20. Use DAP and MOP.",
        "Wheat": "Recommended NPK ratio: 100:50:30. Use Urea, DAP, and MOP.",
        "Paddy": "Recommended NPK ratio: 100:50:50. Use Urea, SSP, and MOP.",
    },
    "Red and Yellow Soil": {
        "Paddy": "Recommended NPK ratio: 100:50:50. Use Urea, SSP, and MOP.",
        "Groundnut": "Recommended NPK ratio: 20:40:40. Use DAP and MOP.",
        "Maize": "Recommended NPK ratio: 120:60:40. Use Urea, DAP, and MOP.",
    },
    "Laterite Soil": {
        "Cashew": "Recommended NPK ratio: 500:125:125 (grams/tree/year). Use Urea, Rock Phosphate, and MOP.",
        "Rubber": "Recommended NPK ratio: 30:30:30. Use a balanced NPK mixture.",
        "Tea": "Fertilizer application depends on age and pruning cycle. Consult local tea board recommendations.",
    },
    "Arid (Desert) Soil": {
        "Millet": "Recommended NPK ratio: 40:20:0. Use Urea and DAP.",
        "Guar": "Recommended NPK ratio: 20:40:0. Use DAP.",
    },
    "Saline Soil": {
        "Barley": "Recommended NPK ratio: 80:40:20. Use Urea, DAP, and MOP. Ensure good drainage.",
        "Cotton": "Use salt-tolerant varieties. Recommended NPK ratio: 150:60:60. Use Urea, DAP, and MOP.",
    },
    "Peaty (Marshy) Soil": {
        "Paddy": "Requires special water management. Recommended NPK ratio: 80:40:40. Use Urea, SSP, and MOP.",
    },
    "Forest and Mountain Soil": {
        "Apple": "Fertilizer application depends on age and soil tests. Consult local horticultural department.",
        "Maize": "Recommended NPK ratio: 80:40:30. Use Urea, DAP, and MOP.",
    }
}

def get_fertilizer_recommendation(soil_type: str, crop: str) -> str:
    """
    Returns fertilizer recommendation based on soil type and crop.
    """
    if soil_type in FERTILIZER_RECOMMENDATIONS and crop in FERTILIZER_RECOMMENDATIONS[soil_type]:
        return FERTILIZER_RECOMMENDATIONS[soil_type][crop]
    else:
        return "No specific fertilizer recommendation found for this soil type and crop combination. Please consult a local agricultural expert."
