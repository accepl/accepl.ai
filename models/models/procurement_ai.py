import random

def optimize_procurement():
    materials = {
        "steel procurement": "Current steel price: ₹60,000 per ton. Recommended supplier: Tata Steel.",
        "cement procurement": "Cement price: ₹6,500 per cubic meter. Supplier: UltraTech Cement.",
        "cables procurement": "Copper cable price: ₹850 per meter. Supplier: Havells."
    }
    return materials.get(random.choice(list(materials.keys())), "AI is analyzing procurement options.")
