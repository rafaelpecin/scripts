import math

class SafetyStockOptimizer:
    """
    Safety Stock Optimizer for Supply Chain & Operations Research.
    
    Calculates safety stock using the standard formula:
        Safety Stock = Z * σ_demand * sqrt(Lead Time)

    Required Constructor Inputs:
        - lead_time_days (float or int): Lead time in days.
        - demand_std_deviation (float): Standard deviation of daily demand.
        - service_level_factor (float): Z-score for desired service level 
                                        (e.g., 2.05 for 98%).
    """

    def __init__(self, lead_time_days, demand_std_deviation, service_level_factor):
        if lead_time_days <= 0:
            raise ValueError("lead_time_days must be > 0.")
        if demand_std_deviation < 0:
            raise ValueError("demand_std_deviation must be >= 0.")
        if service_level_factor < 0:
            raise ValueError("service_level_factor must be >= 0.")

        self.lead_time_days = float(lead_time_days)
        self.demand_std_deviation = float(demand_std_deviation)
        self.service_level_factor = float(service_level_factor)

    def calculate_safety_stock(self):
        """
        Computes safety stock using the standard formula:
            SS = Z * σ * sqrt(LT)
        """
        return self.service_level_factor * self.demand_std_deviation * math.sqrt(self.lead_time_days)

    def summary(self):
        """
        Returns a dictionary summarizing inputs and calculated safety stock.
        """
        return {
            "lead_time_days": self.lead_time_days,
            "demand_std_deviation": self.demand_std_deviation,
            "service_level_factor_Z": self.service_level_factor,
            "safety_stock_units": self.calculate_safety_stock()
        }


# Example usage:
optimizer = SafetyStockOptimizer(lead_time_days=30, demand_std_deviation=12.5, service_level_factor=2.05)
print(optimizer.calculate_safety_stock())
print(optimizer.summary())

