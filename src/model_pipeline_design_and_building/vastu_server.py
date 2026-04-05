import json

class VastuMCPServer:
    """
    Model Context Protocol (MCP) Server for Vastu Shastra Rules.
    This acts as an isolated micro-service that provides external facts
    and scoring heuristics to the Supervisor Agent, keeping rule-engine logic 
    separate from the predictive Deep Learning pipeline.
    """
    def __init__(self):
        # Basic loaded rules (In production, this would query a database)
        self.rules = {
            "entrance": {
                "East": {"score": 10, "advice": "Highly auspicious, brings health and wealth."},
                "North": {"score": 9, "advice": "Very good, ideal for career growth."},
                "West": {"score": 5, "advice": "Average, brings mixed results."},
                "South": {"score": 2, "advice": "Generally avoided, but manageable with specific corrections."}
            },
            "kitchen": {
                "South-East": {"score": 10, "advice": "Perfect location, represents the fire element."},
                "North-West": {"score": 7, "advice": "Acceptable alternative location."},
                "North-East": {"score": 0, "advice": "Strictly avoid, causes domestic friction."}
            },
            "master_bedroom": {
                "South-West": {"score": 10, "advice": "Ideal for stability and peace."},
                "North-East": {"score": 1, "advice": "Avoid, may lead to health issues."}
            }
        }

    def evaluate_property(self, facing: str, kitchen_loc: str = "Unknown", bedroom_loc: str = "Unknown") -> str:
        """
        Tool exposed via MCP to the A2A Supervisor.
        """
        score = 0
        max_score = 30
        report = []
        
        if facing in self.rules["entrance"]:
            rule = self.rules["entrance"][facing]
            score += rule["score"]
            report.append(f"Entrance ({facing}): {rule['advice']}")
            
        if kitchen_loc in self.rules["kitchen"]:
            rule = self.rules["kitchen"][kitchen_loc]
            score += rule["score"]
            report.append(f"Kitchen ({kitchen_loc}): {rule['advice']}")
            
        if bedroom_loc in self.rules["master_bedroom"]:
            rule = self.rules["master_bedroom"][bedroom_loc]
            score += rule["score"]
            report.append(f"Master Bedroom ({bedroom_loc}): {rule['advice']}")
            
        compliance_pct = (score / max_score) * 100 if score > 0 else 50.0 # Default midline
        
        response = {
            "vastu_compliance_percentage": round(compliance_pct, 1),
            "detailed_report": report
        }
        return json.dumps(response, indent=2)

if __name__ == "__main__":
    server = VastuMCPServer()
    result = server.evaluate_property(facing="East", kitchen_loc="South-East", bedroom_loc="South-West")
    print("MCP Server Test Response:")
    print(result)
