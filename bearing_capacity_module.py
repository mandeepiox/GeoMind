"""
Bearing Capacity Calculator Module
Calculates ultimate and safe bearing capacity of soil based on predicted properties
Uses Terzaghi's Bearing Capacity Theory and Meyerhof's equations
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class BearingCapacityResult:
    """Results from bearing capacity calculations"""
    ultimate_bearing_capacity: float  # kN/m²
    safe_bearing_capacity: float  # kN/m²
    factor_of_safety: float
    foundation_type: str
    soil_type: str
    depth_factor: float
    shape_factor: float
    inclination_factor: float
    calculation_method: str
    recommendations: list
    
class BearingCapacityCalculator:
    """
    Calculate bearing capacity using standard geotechnical methods
    """
    
    # Bearing capacity factors (Terzaghi's values)
    NC_VALUES = {
        0: 5.7, 5: 7.3, 10: 9.6, 15: 12.9, 20: 17.7,
        25: 25.1, 30: 37.2, 35: 57.8, 40: 95.7, 45: 172.3
    }
    
    NQ_VALUES = {
        0: 1.0, 5: 1.6, 10: 2.7, 15: 4.4, 20: 7.4,
        25: 12.7, 30: 22.5, 35: 41.4, 40: 81.3, 45: 173.3
    }
    
    NGAMMA_VALUES = {
        0: 0.0, 5: 0.5, 10: 1.2, 15: 2.5, 20: 5.0,
        25: 9.7, 30: 19.7, 35: 42.4, 40: 100.4, 45: 297.5
    }
    
    def __init__(self):
        """Initialize bearing capacity calculator"""
        self.default_fos = 3.0  # Default Factor of Safety
    
    def interpolate_bearing_factor(self, values_dict: Dict, angle: float) -> float:
        """
        Interpolate bearing capacity factors for angles not in table
        
        Args:
            values_dict: Dictionary of angle: factor values
            angle: Angle of internal friction (degrees)
        
        Returns:
            Interpolated factor value
        """
        angles = sorted(values_dict.keys())
        
        # If exact match
        if angle in values_dict:
            return values_dict[angle]
        
        # Find surrounding values
        for i in range(len(angles) - 1):
            if angles[i] < angle < angles[i + 1]:
                # Linear interpolation
                x1, x2 = angles[i], angles[i + 1]
                y1, y2 = values_dict[x1], values_dict[x2]
                return y1 + (y2 - y1) * (angle - x1) / (x2 - x1)
        
        # Extrapolation for values outside range
        if angle < angles[0]:
            return values_dict[angles[0]]
        return values_dict[angles[-1]]
    
    def get_bearing_factors(self, phi: float) -> Tuple[float, float, float]:
        """
        Get bearing capacity factors (Nc, Nq, Nγ) for given friction angle
        
        Args:
            phi: Angle of internal friction (degrees)
        
        Returns:
            Tuple of (Nc, Nq, Nγ)
        """
        Nc = self.interpolate_bearing_factor(self.NC_VALUES, phi)
        Nq = self.interpolate_bearing_factor(self.NQ_VALUES, phi)
        Ngamma = self.interpolate_bearing_factor(self.NGAMMA_VALUES, phi)
        
        return Nc, Nq, Ngamma
    
    def calculate_shape_factors(self, length: float, width: float, phi: float) -> Tuple[float, float, float]:
        """
        Calculate shape factors for rectangular/square footings
        
        Args:
            length: Foundation length (m)
            width: Foundation width (m)
            phi: Angle of internal friction (degrees)
        
        Returns:
            Tuple of (Sc, Sq, Sγ)
        """
        # For strip footing (length >> width)
        if length / width > 10:
            return 1.0, 1.0, 1.0
        
        # For square footing
        if abs(length - width) < 0.1:
            Sc = 1.3
            Sq = 1.2
            Sgamma = 0.8
        else:
            # Rectangular footing
            B_L = width / length
            Sc = 1 + 0.2 * B_L
            Sq = 1 + 0.2 * B_L
            Sgamma = 1 - 0.4 * B_L
        
        return Sc, Sq, Sgamma
    
    def calculate_depth_factors(self, depth: float, width: float, phi: float) -> Tuple[float, float, float]:
        """
        Calculate depth factors for embedment
        
        Args:
            depth: Foundation depth (m)
            width: Foundation width (m)
            phi: Angle of internal friction (degrees)
        
        Returns:
            Tuple of (Dc, Dq, Dγ)
        """
        D_B = depth / width if width > 0 else 0
        
        if D_B <= 1:
            Dc = 1 + 0.4 * D_B
            Dq = 1 + 0.4 * D_B
        else:
            Dc = 1 + 0.4 * np.arctan(D_B)
            Dq = 1 + 0.4 * np.arctan(D_B)
        
        Dgamma = 1.0  # Usually taken as 1.0
        
        return Dc, Dq, Dgamma
    
    def calculate_terzaghi_bearing_capacity(
        self,
        cohesion: float,
        phi: float,
        gamma: float,
        width: float,
        depth: float,
        length: float = None,
        foundation_type: str = "square"
    ) -> float:
        """
        Calculate ultimate bearing capacity using Terzaghi's equation
        
        qu = c*Nc*Sc*Dc + γ*Df*Nq*Sq*Dq + 0.5*γ*B*Nγ*Sγ*Dγ
        
        Args:
            cohesion: Soil cohesion (kPa)
            phi: Angle of internal friction (degrees)
            gamma: Unit weight of soil (kN/m³)
            width: Foundation width (m)
            depth: Foundation depth (m)
            length: Foundation length (m), optional
            foundation_type: "strip", "square", or "circular"
        
        Returns:
            Ultimate bearing capacity (kN/m²)
        """
        # Get bearing capacity factors
        Nc, Nq, Ngamma = self.get_bearing_factors(phi)
        
        # Default length for square/circular
        if length is None:
            length = width
        
        # Calculate shape factors
        if foundation_type == "strip":
            Sc, Sq, Sgamma = 1.0, 1.0, 1.0
        elif foundation_type == "circular":
            Sc, Sq, Sgamma = 1.3, 1.2, 0.6
        else:  # square or rectangular
            Sc, Sq, Sgamma = self.calculate_shape_factors(length, width, phi)
        
        # Calculate depth factors
        Dc, Dq, Dgamma = self.calculate_depth_factors(depth, width, phi)
        
        # Terzaghi's bearing capacity equation
        term1 = cohesion * Nc * Sc * Dc
        term2 = gamma * depth * Nq * Sq * Dq
        term3 = 0.5 * gamma * width * Ngamma * Sgamma * Dgamma
        
        qu = term1 + term2 + term3
        
        return qu
    
    def calculate_meyerhof_bearing_capacity(
        self,
        cohesion: float,
        phi: float,
        gamma: float,
        width: float,
        depth: float,
        length: float = None
    ) -> float:
        """
        Calculate bearing capacity using Meyerhof's equation (more accurate)
        
        Args:
            cohesion: Soil cohesion (kPa)
            phi: Angle of internal friction (degrees)
            gamma: Unit weight of soil (kN/m³)
            width: Foundation width (m)
            depth: Foundation depth (m)
            length: Foundation length (m)
        
        Returns:
            Ultimate bearing capacity (kN/m²)
        """
        if length is None:
            length = width
        
        # Meyerhof's bearing capacity factors
        phi_rad = np.radians(phi)
        Nq = np.exp(np.pi * np.tan(phi_rad)) * (np.tan(np.radians(45 + phi/2)))**2
        Nc = (Nq - 1) / np.tan(phi_rad) if phi > 0 else 5.14
        Ngamma = (Nq - 1) * np.tan(1.4 * phi_rad)
        
        # Shape factors (Meyerhof)
        B_L = width / length
        Sc = 1 + 0.2 * (width / length) * (Nq / Nc)
        Sq = 1 + 0.2 * (width / length) * np.tan(phi_rad)
        Sgamma = 1 - 0.4 * (width / length)
        
        # Depth factors (Meyerhof)
        D_B = depth / width
        Dc = 1 + 0.2 * np.sqrt(Nq) * D_B
        Dq = 1 + 0.1 * np.sqrt(Nq) * D_B
        Dgamma = 1.0
        
        # Calculate bearing capacity
        term1 = cohesion * Nc * Sc * Dc
        term2 = gamma * depth * Nq * Sq * Dq
        term3 = 0.5 * gamma * width * Ngamma * Sgamma * Dgamma
        
        qu = term1 + term2 + term3
        
        return qu
    
    def determine_soil_type_from_properties(
        self,
        cohesion: float,
        phi: float,
        sand_percent: float = None
    ) -> str:
        """
        Determine soil type based on properties
        
        Args:
            cohesion: Cohesion value (kPa)
            phi: Friction angle (degrees)
            sand_percent: Percentage of sand (optional)
        
        Returns:
            Soil type classification
        """
        if cohesion > 50 and phi < 15:
            return "Clayey Soil (High Cohesion)"
        elif cohesion < 10 and phi > 30:
            return "Sandy Soil (High Friction)"
        elif 10 <= cohesion <= 50 and 15 <= phi <= 30:
            return "Silty Soil (Mixed)"
        elif sand_percent:
            if sand_percent > 70:
                return "Sandy Soil"
            elif sand_percent < 30:
                return "Clayey Soil"
            else:
                return "Loamy Soil"
        else:
            return "Mixed Soil"
    
    def get_recommendations(
        self,
        soil_type: str,
        safe_bearing_capacity: float,
        phi: float,
        cohesion: float
    ) -> list:
        """
        Generate engineering recommendations based on bearing capacity
        
        Args:
            soil_type: Type of soil
            safe_bearing_capacity: Safe bearing capacity (kN/m²)
            phi: Friction angle
            cohesion: Cohesion
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Based on bearing capacity
        if safe_bearing_capacity < 100:
            recommendations.append("⚠️ Low bearing capacity - Consider soil improvement or deep foundations")
            recommendations.append("💡 Recommend: Pile foundation or soil stabilization")
        elif 100 <= safe_bearing_capacity < 200:
            recommendations.append("⚠️ Moderate bearing capacity - Suitable for light structures")
            recommendations.append("💡 Recommend: Spread footings with proper design")
        elif 200 <= safe_bearing_capacity < 300:
            recommendations.append("✓ Good bearing capacity - Suitable for most residential structures")
            recommendations.append("💡 Recommend: Isolated or combined footings")
        else:
            recommendations.append("✓✓ Excellent bearing capacity - Suitable for heavy structures")
            recommendations.append("💡 Recommend: Any type of shallow foundation")
        
        # Based on soil type
        if "Clay" in soil_type:
            recommendations.append("⚠️ Clayey soil - Check for settlement and consolidation")
            recommendations.append("💡 Monitor: Long-term settlement over time")
        elif "Sandy" in soil_type:
            recommendations.append("✓ Sandy soil - Good drainage, less settlement issues")
            recommendations.append("💡 Monitor: Density and compaction")
        
        # Based on friction angle
        if phi < 20:
            recommendations.append("⚠️ Low friction angle - May require wider footings")
        elif phi > 35:
            recommendations.append("✓ High friction angle - Good load distribution")
        
        return recommendations
    
    def calculate_bearing_capacity(
        self,
        cohesion: float,
        phi: float,
        bulk_density: float,
        width: float = 1.5,
        depth: float = 1.5,
        length: float = None,
        foundation_type: str = "square",
        factor_of_safety: float = 3.0,
        method: str = "terzaghi",
        sand_percent: float = None
    ) -> BearingCapacityResult:
        """
        Main function to calculate bearing capacity
        
        Args:
            cohesion: Soil cohesion (kPa)
            phi: Angle of internal friction (degrees)
            bulk_density: Bulk density (g/cm³)
            width: Foundation width (m), default 1.5m
            depth: Foundation depth (m), default 1.5m
            length: Foundation length (m)
            foundation_type: "strip", "square", or "circular"
            factor_of_safety: Safety factor, default 3.0
            method: "terzaghi" or "meyerhof"
            sand_percent: Sand percentage for classification
        
        Returns:
            BearingCapacityResult object
        """
        # Convert bulk density from g/cm³ to kN/m³
        gamma = bulk_density * 9.81
        
        # Calculate ultimate bearing capacity
        if method.lower() == "meyerhof":
            qu = self.calculate_meyerhof_bearing_capacity(
                cohesion, phi, gamma, width, depth, length
            )
            calc_method = "Meyerhof's Method"
        else:
            qu = self.calculate_terzaghi_bearing_capacity(
                cohesion, phi, gamma, width, depth, length, foundation_type
            )
            calc_method = "Terzaghi's Method"
        
        # Calculate safe bearing capacity
        qa = qu / factor_of_safety
        
        # Determine soil type
        soil_type = self.determine_soil_type_from_properties(cohesion, phi, sand_percent)
        
        # Get shape and depth factors
        if length is None:
            length = width
        Sc, Sq, Sgamma = self.calculate_shape_factors(length, width, phi)
        Dc, Dq, Dgamma = self.calculate_depth_factors(depth, width, phi)
        
        # Generate recommendations
        recommendations = self.get_recommendations(soil_type, qa, phi, cohesion)
        
        return BearingCapacityResult(
            ultimate_bearing_capacity=round(qu, 2),
            safe_bearing_capacity=round(qa, 2),
            factor_of_safety=factor_of_safety,
            foundation_type=foundation_type,
            soil_type=soil_type,
            depth_factor=round((Dc + Dq + Dgamma) / 3, 3),
            shape_factor=round((Sc + Sq + Sgamma) / 3, 3),
            inclination_factor=1.0,  # Assuming no load inclination
            calculation_method=calc_method,
            recommendations=recommendations
        )


# Utility functions for easy integration
def calculate_bearing_capacity_from_prediction(
    predictions: Dict,
    width: float = 1.5,
    depth: float = 1.5,
    foundation_type: str = "square"
) -> Dict:
    """
    Calculate bearing capacity from soil prediction results
    
    Args:
        predictions: Dictionary of soil property predictions
        width: Foundation width (m)
        depth: Foundation depth (m)
        foundation_type: Type of foundation
    
    Returns:
        Dictionary with bearing capacity results
    """
    calculator = BearingCapacityCalculator()
    
    # Extract required properties
    cohesion = predictions.get('Cohesion', {}).get('value', 20.0)
    phi = predictions.get('Shear angle', {}).get('value', 30.0)
    bulk_density = predictions.get('Bulk Density', {}).get('value', 1.8)
    sand_percent = predictions.get('Sand', {}).get('value', None)
    
    # Calculate bearing capacity
    result = calculator.calculate_bearing_capacity(
        cohesion=cohesion,
        phi=phi,
        bulk_density=bulk_density,
        width=width,
        depth=depth,
        foundation_type=foundation_type,
        sand_percent=sand_percent
    )
    
    return {
        'ultimate_bearing_capacity': result.ultimate_bearing_capacity,
        'safe_bearing_capacity': result.safe_bearing_capacity,
        'unit': 'kN/m²',
        'factor_of_safety': result.factor_of_safety,
        'foundation_type': result.foundation_type,
        'soil_type': result.soil_type,
        'calculation_method': result.calculation_method,
        'recommendations': result.recommendations,
        'factors': {
            'depth_factor': result.depth_factor,
            'shape_factor': result.shape_factor,
            'inclination_factor': result.inclination_factor
        }
    }


# Example usage
if __name__ == "__main__":
    calculator = BearingCapacityCalculator()
    
    # Example: Sandy soil
    result = calculator.calculate_bearing_capacity(
        cohesion=5.0,          # kPa
        phi=35.0,              # degrees
        bulk_density=1.75,     # g/cm³
        width=2.0,             # m
        depth=1.5,             # m
        foundation_type="square",
        sand_percent=70.0
    )
    
    print("=== Bearing Capacity Analysis ===")
    print(f"Soil Type: {result.soil_type}")
    print(f"Ultimate Bearing Capacity: {result.ultimate_bearing_capacity} kN/m²")
    print(f"Safe Bearing Capacity: {result.safe_bearing_capacity} kN/m²")
    print(f"Factor of Safety: {result.factor_of_safety}")
    print(f"Method: {result.calculation_method}")
    print(f"\nRecommendations:")
    for rec in result.recommendations:
        print(f"  {rec}")
