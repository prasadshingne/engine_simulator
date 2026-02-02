"""Engine geometry calculations."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class GeometryParams:
    """Engine geometry parameters."""
    bore: float           # Bore diameter [m]
    stroke: float         # Stroke length [m]
    con_rod: float       # Connecting rod length [m]
    comp_ratio: float    # Geometric compression ratio [-]
    
    def __post_init__(self):
        """Calculate derived parameters."""
        self.crank_radius = self.stroke / 2
        self.bore_area = np.pi * (self.bore/2)**2
        self.displacement = self.bore_area * self.stroke
        self.clearance_volume = self.displacement / (self.comp_ratio - 1)
        
    def piston_position(self, crank_angle: float) -> float:
        """
        Calculate instantaneous piston position from TDC.
        
        Parameters
        ----------
        crank_angle : float
            Crank angle [rad]
            
        Returns
        -------
        float
            Piston position from TDC [m]
        """
        return (self.crank_radius * np.cos(crank_angle) + 
                np.sqrt(self.con_rod**2 - 
                       (self.crank_radius * np.sin(crank_angle))**2))
    
    def cylinder_volume(self, crank_angle: float) -> float:
        """
        Calculate instantaneous cylinder volume.
        
        Parameters
        ----------
        crank_angle : float
            Crank angle [rad]
            
        Returns
        -------
        float
            Cylinder volume [m³]
        """
        x = self.piston_position(crank_angle)
        return self.clearance_volume + self.bore_area * (
            self.con_rod + self.crank_radius - x)
    
    def volume_rate(self, crank_angle: float, rpm: float) -> float:
        """
        Calculate rate of volume change.
        
        Parameters
        ----------
        crank_angle : float
            Crank angle [rad]
        rpm : float
            Engine speed [rev/min]
            
        Returns
        -------
        float
            Rate of volume change [m³/s]
        """
        omega = rpm * 2 * np.pi / 60  # [rad/s]
        
        # Calculate piston velocity
        dx_dt = -self.crank_radius * np.sin(crank_angle) * (
            1 + self.crank_radius * np.cos(crank_angle) / 
            np.sqrt(self.con_rod**2 - 
                   (self.crank_radius * np.sin(crank_angle))**2)
        ) * omega
        
        return self.bore_area * dx_dt
    
    def surface_area(self, crank_angle: float) -> Tuple[float, float, float]:
        """
        Calculate instantaneous surface areas.
        
        Parameters
        ----------
        crank_angle : float
            Crank angle [rad]
            
        Returns
        -------
        Tuple[float, float, float]
            Head area [m²], piston area [m²], liner area [m²]
        """
        # Head and piston areas are constant
        head_area = self.bore_area
        piston_area = self.bore_area
        
        # Liner area varies with piston position
        x = self.piston_position(crank_angle)
        liner_area = np.pi * self.bore * (
            self.con_rod + self.crank_radius - x)
        
        return head_area, piston_area, liner_area 