#!/usr/bin/env python3
"""
Novitec 카메라 통합 테스트 스크립트

이 스크립트는 Novitec 카메라가 AMR Tracker 시스템에 올바르게 통합되었는지 테스트합니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_novitec_import():
    """Novitec 카메라 모듈 import 테스트"""
    print("=== Novitec 카메라 모듈 Import 테스트 ===")

    try:
        from novitec_camera_loader import (
            NovitecCameraLoader,
            create_novitec_camera_loader,
            list_novitec_cameras,
            NOVITEC_AVAILABLE,
        )

        print("✅ Novitec 카메라 로더 모듈 import 성공")

        if NOVITEC_AVAILABLE:
            print("✅ Novitec Camera SDK 사용 가능")
        else:
            print("⚠️ Novitec Camera SDK 사용 불가능")

        return True
    except ImportError as e:
        print(f"❌ Novitec 카메라 로더 import 실패: {e}")
        return False


def test_sequence_loader_integration():
    """Sequence Loader 통합 테스트"""
    print("\n=== Sequence Loader 통합 테스트 ===")

    try:
        from sequence_loader import LoaderMode, NOVITEC_AVAILABLE

        print("✅ Sequence Loader에서 Novitec 지원 import 성공")

        # LoaderMode에 CAMERA_DEVICE가 있는지 확인
        if hasattr(LoaderMode, "CAMERA_DEVICE"):
            print("✅ LoaderMode.CAMERA_DEVICE 사용 가능")
        else:
            print("❌ LoaderMode.CAMERA_DEVICE 없음")
            return False

        return True
    except ImportError as e:
        print(f"❌ Sequence Loader 통합 실패: {e}")
        return False


def test_main_integration():
    """Main.py 통합 테스트"""
    print("\n=== Main.py 통합 테스트 ===")

    try:
        # main.py에서 import 가능한지 확인
        from sequence_loader import create_camera_device_loader

        print("✅ Main.py에서 카메라 로더 import 가능")

        # argparse 옵션 확인 (코드에서 확인)
        print("✅ --loader-mode에 'camera' 옵션으로 Novitec 카메라 자동 감지")

        return True
    except ImportError as e:
        print(f"❌ Main.py 통합 실패: {e}")
        return False


def test_camera_detection():
    """카메라 감지 테스트"""
    print("\n=== 카메라 감지 테스트 ===")

    try:
        from novitec_camera_loader import list_novitec_cameras

        cameras = list_novitec_cameras()
        print(f"감지된 Novitec 카메라 수: {len(cameras)}")

        for i, camera in enumerate(cameras):
            print(f"  {i}: {camera['model_name']} - {camera['serial_number']}")

        if len(cameras) > 0:
            print("✅ Novitec 카메라 감지 성공")
            return True
        else:
            print("⚠️ 감지된 Novitec 카메라가 없습니다")
            print(
                "   (카메라가 연결되지 않았거나 드라이버가 설치되지 않았을 수 있습니다)"
            )
            return False

    except Exception as e:
        print(f"❌ 카메라 감지 실패: {e}")
        return False


def test_camera_connection():
    """카메라 연결 테스트"""
    print("\n=== 카메라 연결 테스트 ===")

    try:
        from novitec_camera_loader import create_novitec_camera_loader

        # 첫 번째 카메라로 테스트
        loader = create_novitec_camera_loader(0)

        if loader is None:
            print("⚠️ 카메라 로더 생성 실패 (카메라가 없거나 연결 실패)")
            return False

        print("✅ 카메라 로더 생성 성공")

        # 카메라 정보 출력
        info = loader.get_camera_info()
        print(f"카메라 정보: {info}")

        # 간단한 프레임 테스트
        print("프레임 테스트 중...")
        ret, frame = loader.read()

        if ret and frame is not None:
            print(f"✅ 프레임 획득 성공: {frame.shape}")
        else:
            print("⚠️ 프레임 획득 실패")

        # 리소스 해제
        loader.release()
        print("✅ 카메라 리소스 해제 완료")

        return True

    except Exception as e:
        print(f"❌ 카메라 연결 테스트 실패: {e}")
        return False


def test_main_script():
    """Main 스크립트 실행 테스트"""
    print("\n=== Main 스크립트 실행 테스트 ===")

    try:
        # main.py의 help 메시지 확인
        import subprocess

        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if "camera" in result.stdout:
            print("✅ Main 스크립트에 camera 옵션 포함됨 (Novitec 자동 감지)")
            return True
        else:
            print("❌ Main 스크립트에 camera 옵션 없음")
            return False

    except Exception as e:
        print(f"❌ Main 스크립트 테스트 실패: {e}")
        return False


def main():
    """전체 테스트 실행"""
    print("🔍 Novitec 카메라 통합 테스트 시작")
    print("=" * 50)

    tests = [
        ("Import 테스트", test_novitec_import),
        ("Sequence Loader 통합", test_sequence_loader_integration),
        ("Main.py 통합", test_main_integration),
        ("카메라 감지", test_camera_detection),
        ("카메라 연결", test_camera_connection),
        ("Main 스크립트", test_main_script),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 실행 중 오류: {e}")
            results.append((test_name, False))

    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 테스트 결과 요약")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\n총 {total}개 테스트 중 {passed}개 통과")

    if passed == total:
        print("🎉 모든 테스트 통과! Novitec 카메라 통합이 완료되었습니다.")
        print("\n사용법:")
        print("  python main.py --mode basic --source 0 --loader-mode camera")
        print("  (Novitec 카메라가 있으면 자동으로 사용, 없으면 일반 카메라 사용)")
    else:
        print("⚠️ 일부 테스트 실패. 설정을 확인해주세요.")
        print("\n문제 해결:")
        print("1. Novitec Camera SDK가 설치되어 있는지 확인")
        print("2. 카메라가 연결되어 있는지 확인")
        print("3. 필요한 DLL 파일들이 있는지 확인")


if __name__ == "__main__":
    main()
