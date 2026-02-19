"""Test script for CALC RESULT (cmd: 6) and MANUAL CALC RESULT (cmd: 9).

Usage:
    # Trajectory analysis (cmd: 6)
    python test_calc_result.py
    python test_calc_result.py --csv data/20260219-112921_zoom1_raw_data.csv
    python test_calc_result.py --csv data/20260219-112921_zoom1_raw_data.csv --sampling 50.0

    # Manual measurement analysis (cmd: 9)
    python test_calc_result.py --manual
    python test_calc_result.py --manual --csv data/20260219-145717_zoom1_start_cam_1_manual_raw_data.csv
"""

import socket
import json
import sys
import argparse
import logging

DEFAULT_CSV = "data/20260219-112921_zoom1_raw_data.csv"
DEFAULT_MANUAL_CSV = "data/20260219-145717_zoom1_start_cam_1_manual_raw_data.csv"


def setup_logger():
    logger = logging.getLogger("test_calc")
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def recv_json(sock) -> dict:
    """Receive a complete JSON response from the socket."""
    buffer = b""
    sock.settimeout(60)

    while True:
        chunk = sock.recv(8192)
        if not chunk:
            raise ConnectionError("Server closed connection")
        buffer += chunk

        text = buffer.decode('utf-8', errors='replace')
        brace_count = 0
        json_start = -1

        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    json_start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and json_start >= 0:
                    json_str = text[json_start:i + 1]
                    response = json.loads(json_str)
                    if response.get("cmd") == 7:
                        buffer = text[i + 1:].encode('utf-8')
                        json_start = -1
                        continue
                    return response


def print_response_data(logger, response):
    """Pretty-print response data and statistics."""
    data = response.get("data", {})
    for key, val in data.items():
        if key == "statistics":
            continue
        logger.info(f"  {key}: {val}")

    stats = data.get("statistics", {})
    if stats:
        logger.info("\n  ── Statistics ──")
        for name, s in stats.items():
            if isinstance(s, dict):
                logger.info(f"    [{name}]")
                for k, v in s.items():
                    if isinstance(v, float):
                        logger.info(f"      {k}: {v:.6f}")
                    else:
                        logger.info(f"      {k}: {v}")


def run_calc_result(sock, logger, csv_path, sampling):
    """Run CALC RESULT (cmd: 6) - trajectory repeatability analysis."""
    request = {
        "cmd": 6,
        "path_csv": csv_path,
        "sampling_interval_mm": sampling,
    }
    logger.info(f"[2/3] CALC RESULT (cmd:6) → path_csv={csv_path}, sampling={sampling}mm")
    sock.sendall(json.dumps(request).encode('utf-8'))
    response = recv_json(sock)

    if response.get("success"):
        logger.info("  RESULT: SUCCESS")
        print_response_data(logger, response)
    else:
        logger.error(f"  RESULT: FAILED")
        logger.error(f"  error_code: {response.get('error_code')}")
        logger.error(f"  error_desc: {response.get('error_desc')}")
        return False
    return True


def run_manual_calc_result(sock, logger, csv_path):
    """Run MANUAL CALC RESULT (cmd: 9) - manual measurement analysis."""
    request = {
        "cmd": 9,
        "path_csv": csv_path,
    }
    logger.info(f"[2/3] MANUAL CALC RESULT (cmd:9) → path_csv={csv_path}")
    sock.sendall(json.dumps(request).encode('utf-8'))
    response = recv_json(sock)

    if response.get("success"):
        logger.info("  RESULT: SUCCESS")
        print_response_data(logger, response)
    else:
        logger.error(f"  RESULT: FAILED")
        logger.error(f"  error_code: {response.get('error_code')}")
        logger.error(f"  error_desc: {response.get('error_desc')}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Test CALC RESULT / MANUAL CALC RESULT")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--csv", default=None, help="CSV file path to analyze")
    parser.add_argument("--sampling", type=float, default=20.0, help="Sampling interval mm (cmd:6 only)")
    parser.add_argument("--manual", action="store_true", help="Use MANUAL CALC RESULT (cmd:9) instead of cmd:6")
    args = parser.parse_args()

    if args.csv is None:
        args.csv = DEFAULT_MANUAL_CSV if args.manual else DEFAULT_CSV

    logger = setup_logger()

    mode = "MANUAL CALC RESULT (cmd:9)" if args.manual else "CALC RESULT (cmd:6)"
    logger.info("=" * 60)
    logger.info(f"Test: {mode}")
    logger.info(f"  Server : {args.host}:{args.port}")
    logger.info(f"  CSV    : {args.csv}")
    if not args.manual:
        logger.info(f"  Sampling: {args.sampling} mm")
    logger.info("=" * 60)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((args.host, args.port))
        logger.info(f"Connected to {args.host}:{args.port}")

        # 1) START VISION (cmd: 1)
        request = {"cmd": 1, "model": "zoom1", "use_area_scan": False}
        logger.info(f"[1/3] START VISION → model=zoom1")
        sock.sendall(json.dumps(request).encode('utf-8'))
        response = recv_json(sock)
        logger.info(f"  Response: success={response.get('success')}")
        if not response.get("success"):
            logger.error(f"  START VISION failed: {response.get('error_desc')}")
            return 1

        # 2) CALC or MANUAL CALC
        if args.manual:
            ok = run_manual_calc_result(sock, logger, args.csv)
        else:
            ok = run_calc_result(sock, logger, args.csv, args.sampling)

        if not ok:
            return 1

        # 3) END VISION (cmd: 2)
        request = {"cmd": 2}
        logger.info(f"[3/3] END VISION")
        sock.sendall(json.dumps(request).encode('utf-8'))
        response = recv_json(sock)
        logger.info(f"  Response: success={response.get('success')}")

        logger.info("\n" + "=" * 60)
        logger.info("Test completed successfully!")
        logger.info("=" * 60)
        return 0

    except ConnectionRefusedError:
        logger.error(f"Connection refused. Is the server running on {args.host}:{args.port}?")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        sock.close()
        logger.info("Disconnected")


if __name__ == "__main__":
    sys.exit(main())
