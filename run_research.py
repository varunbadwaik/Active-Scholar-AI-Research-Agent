
import asyncio
import logging
import sys
from pprint import pprint

# Configure logging to show progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Adjust path if needed
sys.path.append('.')

from state import ResearchState
from graph import graph

async def run():
    print("Starting manual research job: 'Future of AI Agents' (Scope: Narrow)...")
    
    initial_state = {
        "topic": "Future of AI Agents",
        "scope": "narrow",
        "constraints": ["focus on 2025 trends"],
        "max_search_rounds": 1,
        "search_queries": [],
        "search_results": [],
        "explored_urls": set(),
        "current_search_round": 0,
        "search_exhausted": False,
        "ingested_chunks": [],
        "retrieval_results": [],
        "claims": [],
        "contradictions": [],
        "evidence_quality": {},
        "synthesis_complete": False,
        "needs_more_info": False,
        "follow_up_queries": [],
        "report": None,
        "report_metadata": None,
    }

    print("Invoking LangGraph pipeline...")
    try:
        # Use ainvoke for async execution
        final_state = await graph.ainvoke(initial_state)
        
        print("\n" + "="*40)
        print("RESEARCH COMPLETE")
        print("="*40)
        
        report = final_state.get("report")
        if report:
            print("\n=== FINAL REPORT ===\n")
            print(report)
        else:
            print("\nWARNING: No report generated.")
            
        print("\n=== METADATA ===")
        meta = final_state.get("report_metadata", {})
        pprint(meta)
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run())
