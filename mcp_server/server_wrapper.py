#!/usr/bin/env python3
"""
Server wrapper to handle ASGI issues and provide better error handling.
"""

import asyncio
import logging
import signal
import sys
import traceback
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


class GracefulKiller:
    """Handle graceful shutdown on SIGTERM and SIGINT."""
    
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        logger.info(f'Received signal {signum}, initiating graceful shutdown...')
        self.kill_now = True


@asynccontextmanager
async def error_handler():
    """Context manager for handling server errors gracefully."""
    try:
        yield
    except KeyboardInterrupt:
        logger.info('Received keyboard interrupt')
    except Exception as e:
        logger.error(f'Server error: {str(e)}')
        logger.error(f'Traceback: {traceback.format_exc()}')
        raise


async def run_server_with_retry(max_retries=3, retry_delay=5):
    """Run the server with retry logic for handling ASGI errors."""
    killer = GracefulKiller()
    
    for attempt in range(max_retries):
        if killer.kill_now:
            logger.info('Shutdown requested, stopping retry attempts')
            break
            
        try:
            logger.info(f'Starting server attempt {attempt + 1}/{max_retries}')
            
            # Import and run the memory enhanced server
            from memory_enhanced_server import main
            
            async with error_handler():
                await main()
                
            # If we get here, the server shut down normally
            logger.info('Server shut down normally')
            break
            
        except Exception as e:
            logger.error(f'Server attempt {attempt + 1} failed: {str(e)}')
            
            if attempt < max_retries - 1:
                logger.info(f'Retrying in {retry_delay} seconds...')
                await asyncio.sleep(retry_delay)
            else:
                logger.error('Max retries exceeded, giving up')
                raise
                
        if killer.kill_now:
            logger.info('Shutdown requested during retry delay')
            break


async def main():
    """Main entry point with enhanced error handling."""
    try:
        await run_server_with_retry()
    except Exception as e:
        logger.error(f'Fatal error: {str(e)}')
        sys.exit(1)
    finally:
        logger.info('Server wrapper shutting down')


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info('Interrupted by user')
        sys.exit(0)
    except Exception as e:
        logger.error(f'Unexpected error: {str(e)}')
        sys.exit(1)
