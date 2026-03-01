import json
from src.mcp_server.vastu_server import VastuMCPServer

class SupervisorAgent:
    """
    Agent-to-Agent (A2A) Orchestrator.
    This component receives the user's plain text query and coordinates:
    1. The Deep Learning Fusion Model (for ROI / Premium Scores).
    2. The Vastu MCP Server (for traditional architectural analytics).
    """
    def __init__(self):
        self.vastu_mcp = VastuMCPServer()
        # In full production we'd load the PyTorch fusion model here:
        # self.fusion_model = MultimodalFusionModel()
        # self.fusion_model.load_state_dict(...)
        
    def process_query(self, query: str, property_features: dict) -> str:
        """
        Interprets user constraints, pulls data from sub-models, 
        and formulates a synthesized natural language recommendation.
        """
        print(f"Supervisor parsing query: '{query}'")
        
        # 1. Retrieve objective rule-based data via MCP Sub-Agent
        vastu_facing = property_features.get('Facing', 'Unknown')
        mcp_response = self.vastu_mcp.evaluate_property(
            facing=vastu_facing,
            kitchen_loc=property_features.get('Kitchen', 'Unknown'),
            bedroom_loc=property_features.get('Bedroom', 'Unknown')
        )
        vastu_data = json.loads(mcp_response)
        
        # 2. Retrieve quantitative predictions via Deep Learning Agent 
        # (Mocked for immediate architecture testing)
        predicted_3yr_roi = 18.5 # self.fusion_model(property_features...)[0]
        predicted_5yr_roi = 29.2 # self.fusion_model(property_features...)[1]
        visual_premium = 84.0    # self.vision_branch(image)
        
        # 3. Synthesize Final Report
        report = f"### 🤖 Supervisor Agent Recommendation\n\n"
        report += f"**ROI Prediction:** Based on our Multimodal Transformer analysis, this property shows strong potential with an estimated **{predicted_3yr_roi}% ROI over 3 years** and **{predicted_5yr_roi}% over 5 years**.\n\n"
        
        report += f"**Aesthetic Valuation:** The Vision Transformer rated the uploaded images at a **{visual_premium}/100 Premium Score**, indicating high-quality finishes and lighting.\n\n"
        
        report += f"**Vastu Compliance:** Analyzed via MCP Server ({vastu_data['vastu_compliance_percentage']}% Match)\n"
        for detail in vastu_data['detailed_report']:
            report += f"- {detail}\n"
            
        return report

if __name__ == "__main__":
    agent = SupervisorAgent()
    
    test_query = "Is this East-facing apartment a good 5-year investment?"
    test_features = {"Facing": "East", "Kitchen": "South-East", "Bedroom": "South-West"}
    
    response = agent.process_query(test_query, test_features)
    print("\n" + response)
