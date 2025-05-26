#!/usr/bin/env python3
"""
Script to aggregate and analyze results from the GitHub Actions matrix workflow
"""
import json
import os
import glob
from typing import Dict, List, Any
import statistics

def load_json_files(pattern: str) -> List[Dict[str, Any]]:
    """Load all JSON files matching the pattern"""
    files = glob.glob(pattern)
    results = []
    
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                data['_source_file'] = file_path
                results.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return results

def analyze_enhanced_rag_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze enhanced RAG matrix results"""
    analysis = {
        "strategy_performance": {},
        "chunk_size_analysis": {},
        "overall_stats": {}
    }
    
    # Group by strategy
    by_strategy = {}
    by_chunk_size = {}
    
    for result in results:
        strategy = result.get('strategy', 'unknown')
        chunk_size = result.get('chunk_size', 0)
        
        if strategy not in by_strategy:
            by_strategy[strategy] = []
        by_strategy[strategy].append(result)
        
        if chunk_size not in by_chunk_size:
            by_chunk_size[chunk_size] = []
        by_chunk_size[chunk_size].append(result)
    
    # Analyze by strategy
    for strategy, strategy_results in by_strategy.items():
        chunks_created = [r.get('chunks_created', 0) for r in strategy_results]
        search_results = [r.get('search_results', 0) for r in strategy_results]
        
        analysis["strategy_performance"][strategy] = {
            "avg_chunks_created": statistics.mean(chunks_created) if chunks_created else 0,
            "avg_search_results": statistics.mean(search_results) if search_results else 0,
            "total_tests": len(strategy_results),
            "chunk_size_range": [r.get('chunk_size', 0) for r in strategy_results]
        }
    
    # Analyze by chunk size
    for chunk_size, size_results in by_chunk_size.items():
        chunks_created = [r.get('chunks_created', 0) for r in size_results]
        search_results = [r.get('search_results', 0) for r in size_results]
        
        analysis["chunk_size_analysis"][chunk_size] = {
            "avg_chunks_created": statistics.mean(chunks_created) if chunks_created else 0,
            "avg_search_results": statistics.mean(search_results) if search_results else 0,
            "strategies_tested": list(set(r.get('strategy', 'unknown') for r in size_results))
        }
    
    # Overall statistics
    all_chunks = [r.get('chunks_created', 0) for r in results]
    all_search_results = [r.get('search_results', 0) for r in results]
    
    analysis["overall_stats"] = {
        "total_configurations_tested": len(results),
        "avg_chunks_created": statistics.mean(all_chunks) if all_chunks else 0,
        "min_chunks_created": min(all_chunks) if all_chunks else 0,
        "max_chunks_created": max(all_chunks) if all_chunks else 0,
        "avg_search_results": statistics.mean(all_search_results) if all_search_results else 0,
        "strategies_tested": list(by_strategy.keys()),
        "chunk_sizes_tested": list(by_chunk_size.keys())
    }
    
    return analysis

def generate_markdown_report(analysis: Dict[str, Any], unified_results: Dict[str, Any] = None) -> str:
    """Generate a markdown report from the analysis"""
    report = []
    
    report.append("# RAG System Testing Results")
    report.append("")
    report.append("## Matrix Testing Summary")
    report.append("")
    
    if analysis.get("overall_stats"):
        stats = analysis["overall_stats"]
        report.append(f"- **Total Configurations Tested**: {stats['total_configurations_tested']}")
        report.append(f"- **Strategies Tested**: {', '.join(stats['strategies_tested'])}")
        report.append(f"- **Chunk Sizes Tested**: {', '.join(map(str, stats['chunk_sizes_tested']))}")
        report.append(f"- **Average Chunks Created**: {stats['avg_chunks_created']:.1f}")
        report.append(f"- **Chunk Range**: {stats['min_chunks_created']} - {stats['max_chunks_created']}")
        report.append("")
    
    # Strategy Performance
    if analysis.get("strategy_performance"):
        report.append("## Strategy Performance")
        report.append("")
        report.append("| Strategy | Avg Chunks | Avg Results | Tests Run |")
        report.append("|----------|------------|-------------|-----------|")
        
        for strategy, perf in analysis["strategy_performance"].items():
            report.append(f"| {strategy} | {perf['avg_chunks_created']:.1f} | {perf['avg_search_results']:.1f} | {perf['total_tests']} |")
        report.append("")
    
    # Chunk Size Analysis
    if analysis.get("chunk_size_analysis"):
        report.append("## Chunk Size Analysis")
        report.append("")
        report.append("| Chunk Size | Avg Chunks | Avg Results | Strategies |")
        report.append("|------------|------------|-------------|------------|")
        
        for size, analysis_data in analysis["chunk_size_analysis"].items():
            strategies = ", ".join(analysis_data['strategies_tested'])
            report.append(f"| {size} | {analysis_data['avg_chunks_created']:.1f} | {analysis_data['avg_search_results']:.1f} | {strategies} |")
        report.append("")
    
    # Unified Workflow Results
    if unified_results:
        report.append("## Unified Workflow Comparison")
        report.append("")
        
        basic = unified_results.get("basic_rag", {})
        enhanced = unified_results.get("enhanced_rag", {})
        comparison = unified_results.get("comparison", {})
        
        if basic and enhanced:
            report.append("### Performance Comparison")
            report.append("")
            report.append("| Metric | Basic RAG | Enhanced RAG | Winner |")
            report.append("|--------|-----------|--------------|--------|")
            
            perf_comp = comparison.get("performance_comparison", {})
            if perf_comp:
                setup_winner = perf_comp.get("setup_time", {}).get("winner", "N/A")
                search_winner = perf_comp.get("search_speed", {}).get("winner", "N/A")
                speedup = perf_comp.get("search_speed", {}).get("speedup", 1)
                
                report.append(f"| Setup Time | {basic.get('setup_time', 0):.3f}s | {enhanced.get('setup_time', 0):.3f}s | {setup_winner} |")
                report.append(f"| Search Speed | {basic.get('avg_search_time', 0):.4f}s | {enhanced.get('avg_search_time', 0):.4f}s | {search_winner} ({speedup:.2f}x) |")
                report.append(f"| Documents/Chunks | {basic.get('total_documents', 0)} docs | {enhanced.get('total_chunks', 0)} chunks | Enhanced (granular) |")
            
            report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    
    if analysis.get("strategy_performance"):
        # Find best performing strategy
        best_strategy = max(
            analysis["strategy_performance"].items(),
            key=lambda x: x[1]["avg_chunks_created"]
        )
        
        report.append(f"- **Best Chunking Strategy**: `{best_strategy[0]}` (avg {best_strategy[1]['avg_chunks_created']:.1f} chunks)")
    
    if analysis.get("chunk_size_analysis"):
        # Find optimal chunk size
        best_size = max(
            analysis["chunk_size_analysis"].items(),
            key=lambda x: x[1]["avg_search_results"]
        )
        
        report.append(f"- **Optimal Chunk Size**: `{best_size[0]}` characters (avg {best_size[1]['avg_search_results']:.1f} results)")
    
    report.append("- **Use Enhanced RAG** for production systems requiring document chunking")
    report.append("- **Use Basic RAG** for simple, small document scenarios")
    report.append("")
    
    return "\n".join(report)

def main():
    """Main function to aggregate and analyze results"""
    print("🔍 Aggregating GitHub Actions matrix results...")
    
    # Load enhanced RAG results
    enhanced_results = load_json_files("enhanced_rag_results_*.json")
    print(f"📊 Found {len(enhanced_results)} enhanced RAG result files")
    
    # Load unified workflow results
    unified_results = {}
    if os.path.exists("unified_rag_results.json"):
        with open("unified_rag_results.json", 'r') as f:
            unified_results = json.load(f)
        print("📈 Found unified workflow results")
    
    # Analyze results
    if enhanced_results:
        analysis = analyze_enhanced_rag_results(enhanced_results)
        
        # Generate report
        report = generate_markdown_report(analysis, unified_results)
        
        # Save report
        with open("matrix_results_report.md", 'w') as f:
            f.write(report)
        
        # Save analysis JSON
        with open("matrix_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print("✅ Analysis complete!")
        print("📄 Report saved to: matrix_results_report.md")
        print("📊 Analysis data saved to: matrix_analysis.json")
        
        # Print summary
        print("\n" + "="*60)
        print("MATRIX TESTING SUMMARY")
        print("="*60)
        
        stats = analysis.get("overall_stats", {})
        print(f"Total configurations tested: {stats.get('total_configurations_tested', 0)}")
        print(f"Strategies tested: {', '.join(stats.get('strategies_tested', []))}")
        print(f"Chunk sizes tested: {', '.join(map(str, stats.get('chunk_sizes_tested', [])))}")
        print(f"Average chunks created: {stats.get('avg_chunks_created', 0):.1f}")
        
        if unified_results:
            print(f"\nUnified workflow comparison available")
            basic_docs = unified_results.get("basic_rag", {}).get("total_documents", 0)
            enhanced_chunks = unified_results.get("enhanced_rag", {}).get("total_chunks", 0)
            print(f"Basic RAG: {basic_docs} documents")
            print(f"Enhanced RAG: {enhanced_chunks} chunks")
    
    else:
        print("❌ No enhanced RAG results found")

if __name__ == "__main__":
    main()
