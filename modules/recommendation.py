"""
Recommendation Module
-----------------------
Maps a predicted deficiency class to personalized dietary guidance.
Corresponds to Chapter 7.6 of the NUTRICARE report.
"""

RECOMMENDATIONS = {
    "Iron": {
        "summary": "Signs consistent with Iron deficiency were detected.",
        "foods": [
            "Leafy greens (spinach, kale)",
            "Legumes (lentils, chickpeas, beans)",
            "Lean red meat and poultry",
            "Fortified cereals",
        ],
        "tips": [
            "Pair iron-rich foods with Vitamin C sources (citrus, tomatoes, bell peppers) to improve absorption.",
            "Avoid tea or coffee immediately after iron-rich meals; tannins can inhibit iron absorption.",
        ],
    },
    "Vitamin_B12": {
        "summary": "Signs consistent with Vitamin B12 deficiency were detected.",
        "foods": [
            "Dairy products (milk, yogurt, cheese)",
            "Eggs",
            "Fortified breakfast cereals",
            "Fish and shellfish",
        ],
        "tips": [
            "Vegetarians and vegans should consider fortified foods or supplementation, as B12 is mainly found in animal products.",
            "Persistent symptoms (fatigue, tingling, memory issues) warrant a clinical blood test.",
        ],
    },
    "Vitamin_D": {
        "summary": "Signs consistent with Vitamin D deficiency were detected.",
        "foods": [
            "Fatty fish (salmon, mackerel, sardines)",
            "Egg yolks",
            "Fortified milk and plant-based milks",
            "Mushrooms exposed to sunlight",
        ],
        "tips": [
            "Aim for regular, safe sun exposure (10-15 minutes a few times a week), skin tone and climate permitting.",
            "Consider a supplement in consultation with a doctor, especially during low-sunlight seasons.",
        ],
    },
    "Protein": {
        "summary": "Signs consistent with Protein deficiency were detected.",
        "foods": [
            "Lean meats, poultry, and fish",
            "Eggs and dairy",
            "Legumes, lentils, and tofu",
            "Nuts and seeds",
        ],
        "tips": [
            "Spread protein intake across meals rather than concentrating it in one sitting.",
            "Combine plant proteins (e.g. rice + beans) to obtain a complete amino-acid profile.",
        ],
    },
    "No_Deficiency": {
        "summary": "No visible signs of the screened deficiencies were detected.",
        "foods": [
            "Continue a balanced, varied diet rich in fruits, vegetables, whole grains, and lean protein.",
        ],
        "tips": [
            "This is a screening result, not a clinical diagnosis. Routine check-ups are still recommended.",
        ],
    },
}


def get_recommendation(predicted_class: str) -> dict:
    """
    Returns the recommendation dict for a predicted class. Falls back to
    a safe default if an unrecognized label is passed in.
    """
    return RECOMMENDATIONS.get(
        predicted_class,
        {
            "summary": "Unable to determine a specific recommendation.",
            "foods": [],
            "tips": ["Please consult a healthcare professional for further evaluation."],
        },
    )
