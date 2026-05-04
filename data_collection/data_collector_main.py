#!/usr/bin/env python3
"""
Main data collection script that orchestrates both Twitter and Yahoo Finance data collection
"""

import os
import sys
import argparse
from datetime import datetime
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from twitter_collector import TwitterCollector
from unified_stock_collector import UnifiedStockCollector
from realtime_event_monitor import RealTimeEventMonitor
from config import STOCK_SYMBOLS, DATA_CONFIG

class DataCollectionOrchestrator:
    def __init__(self):
        """Initialize the data collection orchestrator"""
        self.twitter_collector = TwitterCollector()
        self.stock_collector = UnifiedStockCollector()
        self.event_monitor = RealTimeEventMonitor()
        self.data_dir = DATA_CONFIG['raw_data_dir']
        
    def collect_all_data(self, stock_symbols=None, twitter_enabled=True, stock_enabled=True,
                         start_date: str = None, end_date: str = None, realtime_monitoring=False):
        """
        Collect both Twitter and Stock data using unified collectors
        
        Args:
            stock_symbols: List of stock symbols to collect data for
            twitter_enabled: Whether to collect Twitter data
            stock_enabled: Whether to collect stock data
            start_date: Start date for data collection
            end_date: End date for data collection
            realtime_monitoring: Whether to enable real-time event monitoring
            
        Returns:
            Dictionary with collection results
        """
        if stock_symbols is None:
            stock_symbols = STOCK_SYMBOLS
        
        results = {
            'collection_timestamp': datetime.now().isoformat(),
            'symbols': stock_symbols,
            'twitter_data': None,
            'stock_data': None,
            'realtime_monitoring': realtime_monitoring,
            'status': 'success'
        }
        
        print(f"Starting data collection for symbols: {', '.join(stock_symbols)}")
        if start_date:
            print(f"Date range: {start_date} to {end_date or 'latest'}")
        print(f"Twitter collection: {'Enabled' if twitter_enabled else 'Disabled'}")
        print(f"Stock data collection: {'Enabled' if stock_enabled else 'Disabled'}")
        print(f"Real-time monitoring: {'Enabled' if realtime_monitoring else 'Disabled'}")
        
        # Collect Twitter data
        if twitter_enabled:
            try:
                print("\n" + "="*50)
                print("COLLECTING TWITTER DATA")
                print("="*50)
                
                start_dt = datetime.fromisoformat(start_date) if start_date else None
                end_dt = datetime.fromisoformat(end_date) if end_date else None
                tweets_data = self.twitter_collector.collect_tweets_for_stocks(
                    stock_symbols,
                    start_time=start_dt,
                    end_time=end_dt,
                    max_tweets_per_symbol=2000
                )
                twitter_filepath = self.twitter_collector.save_tweets_to_json(tweets_data)
                twitter_stats = self.twitter_collector.get_tweet_statistics(tweets_data)
                
                results['twitter_data'] = {
                    'filepath': twitter_filepath,
                    'statistics': twitter_stats
                }
                
                print(f"Twitter collection completed successfully!")
                print(f"Total tweets collected: {twitter_stats['total_tweets']}")
                
            except Exception as e:
                print(f"Twitter collection failed: {e}")
                results['twitter_data'] = {'error': str(e)}
                results['status'] = 'partial_failure'
        
        # Collect Stock data using unified collector
        if stock_enabled:
            try:
                print("\n" + "="*50)
                print("COLLECTING STOCK DATA (Unified APIs)")
                print("="*50)
                
                start_dt = datetime.fromisoformat(start_date) if start_date else None
                end_dt = datetime.fromisoformat(end_date) if end_date else None
                stocks_data = self.stock_collector.get_multiple_stocks_data(
                    stock_symbols,
                    period="max" if not start_dt else "1d",
                    interval="1d",
                    start=start_dt,
                    end=end_dt
                )
                combined_data = self.stock_collector.combine_stocks_data(stocks_data)
                enhanced_data = self.stock_collector.calculate_technical_indicators(combined_data)
                stock_filepath = self.stock_collector.save_stock_data_to_csv(enhanced_data)
                stock_stats = self.stock_collector.get_data_statistics(enhanced_data)
                
                results['stock_data'] = {
                    'filepath': stock_filepath,
                    'statistics': stock_stats
                }
                
                print(f"Stock data collection completed successfully!")
                print(f"Total records collected: {stock_stats['total_records']}")
                print(f"APIs used: {', '.join(stock_stats.get('api_usage', {}).keys())}")
                
            except Exception as e:
                print(f"Stock data collection failed: {e}")
                results['stock_data'] = {'error': str(e)}
                results['status'] = 'partial_failure'
        
        # Start real-time monitoring if requested
        if realtime_monitoring:
            try:
                print("\n" + "="*50)
                print("STARTING REAL-TIME EVENT MONITORING")
                print("="*50)
                
                self.event_monitor.start_monitoring()
                results['realtime_monitoring'] = {'status': 'started'}
                print("Real-time monitoring started successfully!")
                
            except Exception as e:
                print(f"Real-time monitoring failed to start: {e}")
                results['realtime_monitoring'] = {'error': str(e)}
                results['status'] = 'partial_failure'
        
        # Save collection summary
        self._save_collection_summary(results)
        
        return results
    
    def _save_collection_summary(self, results):
        """Save collection summary to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_filepath = os.path.join(self.data_dir, f"collection_summary_{timestamp}.json")
        
        with open(summary_filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nCollection summary saved to: {summary_filepath}")
    
    def validate_data_quality(self, results):
        """
        Validate the quality of collected data
        
        Args:
            results: Collection results dictionary
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'twitter_validation': None,
            'yahoo_validation': None,
            'overall_status': 'unknown'
        }
        
        # Validate Twitter data
        if results.get('twitter_data') and 'error' not in results['twitter_data']:
            twitter_stats = results['twitter_data']['statistics']
            twitter_validation = {
                'total_tweets': twitter_stats['total_tweets'],
                'symbols_with_data': len([s for s, stats in twitter_stats['symbol_stats'].items() 
                                        if stats['tweet_count'] > 0]),
                'avg_tweets_per_symbol': twitter_stats['total_tweets'] / len(twitter_stats['symbol_stats']),
                'status': 'good' if twitter_stats['total_tweets'] > 100 else 'insufficient'
            }
            validation_results['twitter_validation'] = twitter_validation
        
        # Validate Stock data
        if results.get('stock_data') and 'error' not in results['stock_data']:
            stock_stats = results['stock_data']['statistics']
            stock_validation = {
                'total_records': stock_stats['total_records'],
                'symbols_with_data': len(stock_stats['symbols']),
                'date_range_days': (datetime.fromisoformat(stock_stats['date_range']['end']) - 
                                  datetime.fromisoformat(stock_stats['date_range']['start'])).days,
                'apis_used': list(stock_stats.get('api_usage', {}).keys()),
                'status': 'good' if stock_stats['total_records'] > 1000 else 'insufficient'
            }
            validation_results['stock_validation'] = stock_validation
        
        # Overall validation status
        twitter_status = validation_results['twitter_validation']['status'] if validation_results['twitter_validation'] else 'unknown'
        stock_status = validation_results['stock_validation']['status'] if validation_results['stock_validation'] else 'unknown'
        
        if twitter_status == 'good' and stock_status == 'good':
            validation_results['overall_status'] = 'excellent'
        elif twitter_status in ['good', 'unknown'] and stock_status in ['good', 'unknown']:
            validation_results['overall_status'] = 'good'
        else:
            validation_results['overall_status'] = 'needs_improvement'
        
        return validation_results

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Collect Twitter and Stock data for sentiment analysis with real-time monitoring')
    parser.add_argument('--symbols', nargs='+', default=STOCK_SYMBOLS, 
                       help='Stock symbols to collect data for')
    parser.add_argument('--no-twitter', action='store_true', 
                       help='Skip Twitter data collection')
    parser.add_argument('--no-stock', action='store_true', 
                       help='Skip stock data collection')
    parser.add_argument('--realtime', action='store_true',
                       help='Enable real-time event monitoring')
    parser.add_argument('--validate', action='store_true', 
                       help='Validate data quality after collection')
    parser.add_argument('--start', type=str, default='2018-01-01',
                       help='ISO date (YYYY-MM-DD) to start collection from (default: 2018-01-01)')
    parser.add_argument('--end', type=str, default=None,
                       help='ISO date (YYYY-MM-DD) to end collection at (default: latest)')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = DataCollectionOrchestrator()
    
    # Collect data
    results = orchestrator.collect_all_data(
        stock_symbols=args.symbols,
        twitter_enabled=not args.no_twitter,
        stock_enabled=not args.no_stock,
        start_date=args.start,
        end_date=args.end,
        realtime_monitoring=args.realtime
    )
    
    # Validate data if requested
    if args.validate:
        print("\n" + "="*50)
        print("DATA VALIDATION")
        print("="*50)
        
        validation_results = orchestrator.validate_data_quality(results)
        
        print(f"Overall validation status: {validation_results['overall_status']}")
        
        if validation_results['twitter_validation']:
            tv = validation_results['twitter_validation']
            print(f"Twitter data: {tv['total_tweets']} tweets, {tv['symbols_with_data']} symbols")
        
        if validation_results['stock_validation']:
            sv = validation_results['stock_validation']
            print(f"Stock data: {sv['total_records']} records, {sv['symbols_with_data']} symbols")
            print(f"APIs used: {', '.join(sv['apis_used'])}")
    
    # Print final summary
    print("\n" + "="*50)
    print("COLLECTION SUMMARY")
    print("="*50)
    print(f"Status: {results['status']}")
    print(f"Symbols processed: {', '.join(results['symbols'])}")
    
    if results['twitter_data'] and 'error' not in results['twitter_data']:
        print(f"Twitter data: {results['twitter_data']['filepath']}")
    
    if results['stock_data'] and 'error' not in results['stock_data']:
        print(f"Stock data: {results['stock_data']['filepath']}")
    
    if results.get('realtime_monitoring') and 'error' not in results['realtime_monitoring']:
        print(f"Real-time monitoring: {results['realtime_monitoring']['status']}")

if __name__ == "__main__":
    main()
