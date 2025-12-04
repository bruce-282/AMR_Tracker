"""Run Vision Tracking TCP/IP Server."""

import argparse
import logging
import os

# Fix OpenMP library conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.server.vision_server import VisionServer


def main():
    """Run vision server."""
    parser = argparse.ArgumentParser(description="Vision Tracking TCP/IP Server")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10000,
        help="Server port (default: 10000)"
    )
    # parser.add_argument(
    #     "--config",
    #     default="tracker_config.json",
    #     help="Configuration file path (default: tracker_config.json)"
    # )
    parser.add_argument(
        "--preset",
        default="camera_tracking",
        help="Preset name to use (e.g., 'camera_tracking', 'video_tracking', 'sequence_tracking'). If not specified, uses default_loader_mode from config."
    )
    
    args = parser.parse_args()
    
    # Setup logger
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s', 
                                    datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("Vision Tracking TCP/IP Server")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    if args.preset:
        logger.info(f"Preset: {args.preset}")
    else:
        logger.info("Preset: None")
    logger.info("=" * 60)
    
    server = VisionServer(
        host=args.host,
        port=args.port,
        preset_name=args.preset
    )
    
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("\nShutting down server...")
        server.stop()
    except Exception as e:
        logger.warning(f"Server error: {e}")
        server.stop()


if __name__ == "__main__":
    main()

