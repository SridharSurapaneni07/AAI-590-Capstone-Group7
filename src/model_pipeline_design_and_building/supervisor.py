import json
from src.model_pipeline_design_and_building.vastu_server import VastuMCPServer

PROPERTY_ENTITIES = ["property", "apartment", "villa", "house", "home", "real estate", "flat", "bhk", "land", "plot"]
PROPERTY_TOPICS = ["investment", "valuation", "price", "roi", "vastu", "facing", "kitchen", "south", "north", "east", "west", "wealth", "buy", "sell", "return", "good", "bad", "worth"]


class SupervisorAgent:
    """
    Agent-to-Agent (A2A) Orchestrator.
    Receives user queries and coordinates between:
    1. The Deep Learning Fusion Model (ROI / Premium Scores)
    2. The Vastu MCP Server (cultural architectural analytics)
    """
    def __init__(self):
        self.vastu_mcp = VastuMCPServer()

    def _is_property_related(self, query: str) -> bool:
        """Check if the query is related to real estate context by requiring an entity and a topic."""
        query_lower = query.lower()
        
        # Immediate match for highly specific analytical queries
        if "vastu" in query_lower or "roi" in query_lower:
            return True
            
        has_entity = any(kw in query_lower for kw in PROPERTY_ENTITIES)
        has_topic = any(kw in query_lower for kw in PROPERTY_TOPICS)
        
        return has_entity and has_topic

    def _extract_facing(self, query: str) -> str:
        """Extract facing direction from the user's query text."""
        query_lower = query.lower()
        directions = {
            "north-east": "North", "north-west": "North",
            "south-east": "South", "south-west": "South",
            "north": "North", "south": "South",
            "east": "East", "west": "West"
        }
        for pattern, direction in directions.items():
            if pattern in query_lower:
                return direction
        return None

    def _extract_kitchen(self, query: str) -> str:
        """Extract kitchen location from query."""
        query_lower = query.lower()
        if "south-east" in query_lower and "kitchen" in query_lower:
            return "South-East"
        if "north-west" in query_lower and "kitchen" in query_lower:
            return "North-West"
        if "north-east" in query_lower and "kitchen" in query_lower:
            return "North-East"
        return None

    def _format_vastu_response(self, query: str, property_features: dict) -> str:
        """Generate a Vastu-focused analysis using the MCP server."""
        facing = property_features.get('Facing') or self._extract_facing(query) or 'Unknown'
        kitchen = property_features.get('Kitchen') or self._extract_kitchen(query) or 'Unknown'
        bedroom = property_features.get('Bedroom', 'Unknown')

        mcp_response = self.vastu_mcp.evaluate_property(
            facing=facing,
            kitchen_loc=kitchen,
            bedroom_loc=bedroom
        )
        vastu_data = json.loads(mcp_response)

        report = f"**Vastu Compliance Analysis**\n\n"
        report += f"**Overall Score:** {vastu_data['vastu_compliance_percentage']}% Vastu Compliant\n\n"

        if vastu_data['detailed_report']:
            report += "**Detailed Assessment:**\n"
            for detail in vastu_data['detailed_report']:
                report += f"- {detail}\n"
        else:
            report += f"*Note: Limited property details available (Facing: {facing}). "
            report += "Provide kitchen and bedroom locations for a complete Vastu assessment.*\n"

        return report

    def _format_investment_response(self, query: str, property_features: dict) -> str:
        """Generate an investment-focused analysis using available property data."""
        report = "**Investment Analysis**\n\n"

        city = property_features.get('City', 'Unknown')
        locality = property_features.get('Locality', 'Unknown')
        price = property_features.get('Price_INR_Cr', 0)
        roi3 = property_features.get('Actual_3Yr_ROI_Pct', 0)
        roi5 = property_features.get('Actual_5Yr_ROI_Pct', 0)
        prop_type = property_features.get('PropertyType', 'Property')
        bhk = property_features.get('BHK', 'N/A')
        facing = property_features.get('Facing', 'Unknown')

        # ROI assessment
        if roi3 and float(roi3) > 0:
            roi3_val = float(roi3)
            roi5_val = float(roi5) if roi5 else 0

            if roi3_val > 25:
                roi_grade = "Excellent"
                roi_advice = "This property shows exceptionally strong short-term appreciation potential."
            elif roi3_val > 15:
                roi_grade = "Good"
                roi_advice = "Solid growth trajectory — competitive within the local market."
            elif roi3_val > 5:
                roi_grade = "Moderate"
                roi_advice = "Average returns. Consider the long-term horizon for better appreciation."
            else:
                roi_grade = "Below Average"
                roi_advice = "Low short-term returns. May be a long-term hold or undervalued asset."

            report += f"- **Property:** {bhk} BHK {prop_type} in {locality}, {city}\n"
            report += f"- **Price:** Rs. {price:.2f} Cr\n"
            report += f"- **Facing:** {facing}\n\n"
            report += f"**ROI Assessment ({roi_grade}):**\n"
            report += f"- 3-Year Projected ROI: **{roi3_val:.1f}%**\n"
            report += f"- 5-Year Projected ROI: **{roi5_val:.1f}%**\n"
            report += f"- *{roi_advice}*\n\n---\n\n"
        else:
            report += f"- **Property:** {prop_type} in {locality}, {city}\n"
            report += f"- **Price:** Rs. {price:.2f} Cr\n\n"
            report += "*ROI projections are not available for this property.*\n\n---\n\n"

        # Add Vastu summary
        report += self._format_vastu_response(query, property_features)

        return report

    def process_query(self, query: str, property_features: dict) -> str:
        """
        Main entry point. Interprets user query, routes to appropriate sub-agents.
        """
        # Out-of-context check
        if not self._is_property_related(query):
            return (
                "I'm a real estate investment advisor specializing in property valuation, "
                "ROI forecasting, and Vastu compliance analysis for Indian real estate.\n\n"
                "You can ask me things like:\n"
                "- *Is this East-facing apartment a good investment?*\n"
                "- *What's the Vastu score for a South-facing villa?*\n"
                "- *Analyze this property for 5-year ROI potential*\n"
                "- *Is North-East kitchen placement good per Vastu?*\n\n"
                "Please ask a property-related question and I'll analyze it for you."
            )

        query_lower = query.lower()

        # Route: Vastu-specific query
        if "vastu" in query_lower or "facing" in query_lower or "kitchen" in query_lower:
            return self._format_vastu_response(query, property_features)

        # Route: Investment / ROI query
        return self._format_investment_response(query, property_features)


if __name__ == "__main__":
    agent = SupervisorAgent()

    # Test 1: Property query
    test_features = {"Facing": "East", "Kitchen": "South-East", "Bedroom": "South-West",
                     "City": "Bengaluru", "Locality": "Whitefield", "Price_INR_Cr": 2.5,
                     "PropertyType": "Apartment", "BHK": 3,
                     "Actual_3Yr_ROI_Pct": 22.5, "Actual_5Yr_ROI_Pct": 35.1}
    print(agent.process_query("Is this apartment good for wealth?", test_features))

    print("\n---\n")

    # Test 2: Out of context
    print(agent.process_query("What is the weather today?", {}))
