import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend (no GUI required)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interp1d
from typing import Tuple, List, Dict
import argparse

from src.utils.csv_writer import MAX_TRAJECTORY_POINTS


class TrajectoryRepeatability:
    """궤적 및 정지 위치 반복정밀도 분석"""

    # Cam1 위치·각도 컬럼 그룹 (이 중 NaN 또는 0.0인 행은 계산에서 제외)
    _POSITION_COLUMN_GROUPS = [
        ["cam_1_x(mm)", "cam_1_y(mm)", "cam_1_rz(deg)"],
    ]

    def __init__(self, csv_path: str):
        """
        CSV 파일을 로드합니다.

        Args:
            csv_path: CSV 파일 경로

        Note:
            CSV 파일의 일부 행이 헤더보다 많은 컬럼을 가질 수 있습니다.
            이 경우 헤더에 정의된 컬럼 수만큼만 읽습니다.
            cam_1_x, cam_1_y, cam_1_rz(및 cam_3 동일) 값이 NaN 또는 0.0인 행은 계산에서 제외합니다.
        """
        # 먼저 헤더만 읽어서 컬럼 수 확인
        try:
            self.df = pd.read_csv(csv_path)
        except pd.errors.ParserError as e:
            print(f"CSV 파싱 오류 감지: {e}")
            print("헤더 컬럼 수에 맞게 데이터를 자르는 중...")

            with open(csv_path, "r", encoding="utf-8") as f:
                header_line = f.readline().strip()
            n_cols = len(header_line.split(","))

            self.df = pd.read_csv(
                csv_path,
                usecols=range(n_cols),
                on_bad_lines="warn",
            )
            print(f"✓ {n_cols}개 컬럼만 로드 완료")

        self._coerce_numeric_columns()
        self.results = {}

    def _coerce_numeric_columns(self):
        """cam_1/cam_2/cam_3 수치 컬럼을 강제 numeric 변환 (공백·문자열 → NaN)."""
        for col in self.df.columns:
            if col.startswith(("cam_1_", "cam_2_", "cam_3_", "velocity")):
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

    def _valid_mask_for_columns(self, cols: List[str]) -> np.ndarray:
        """지정한 컬럼들에 대해 NaN·0.0이 아닌 행만 True. (행은 삭제하지 않고, 계산 시에만 사용.)"""
        if not all(c in self.df.columns for c in cols):
            return np.ones(len(self.df), dtype=bool)
        mask = np.ones(len(self.df), dtype=bool)
        for c in cols:
            try:
                vals = pd.to_numeric(self.df[c], errors="coerce")
            except Exception:
                vals = self.df[c]
            mask &= pd.notna(vals) & (np.asarray(vals, dtype=float) != 0.0)
        return mask

    def circular_mean(self, angles_deg: np.ndarray) -> float:
        """각도의 circular mean 계산 (degree 단위)"""
        angles_rad = np.deg2rad(angles_deg)
        sin_mean = np.mean(np.sin(angles_rad))
        cos_mean = np.mean(np.cos(angles_rad))
        return np.rad2deg(np.arctan2(sin_mean, cos_mean))

    def angular_diff(self, angle1_deg: np.ndarray, angle2_deg: float) -> np.ndarray:
        """두 각도의 shortest angular distance (degree)"""
        diff = angle1_deg - angle2_deg
        # -180 ~ 180 범위로 normalize
        diff = (diff + 180) % 360 - 180
        return diff

    def calculate_arc_length(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """궤적의 누적 arc-length 계산"""
        dx = np.diff(x)
        dy = np.diff(y)
        segment_lengths = np.sqrt(dx**2 + dy**2)
        arc_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        return arc_length

    def interpolate_trajectory(
        self,
        x: np.ndarray,
        y: np.ndarray,
        theta: np.ndarray,
        arc_length: np.ndarray,
        target_s: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Arc-length 기반으로 궤적을 uniform sampling"""
        # x, y는 linear interpolation
        interp_x = interp1d(
            arc_length, x, kind="linear", bounds_error=False, fill_value="extrapolate"
        )
        interp_y = interp1d(
            arc_length, y, kind="linear", bounds_error=False, fill_value="extrapolate"
        )

        # theta는 unwrap 후 interpolation, 다시 wrap
        theta_unwrap = np.unwrap(np.deg2rad(theta))
        interp_theta = interp1d(
            arc_length,
            theta_unwrap,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        x_resampled = interp_x(target_s)
        y_resampled = interp_y(target_s)
        theta_resampled = np.rad2deg(interp_theta(target_s)) % 360

        return x_resampled, y_resampled, theta_resampled

    def analyze_static_position(
        self, cam_name: str, x_col: str, y_col: str, theta_col: str
    ) -> Dict:
        """정지 위치 반복정밀도 계산 (Cam1, Cam3). NaN·0.0 행은 기록은 유지하고 평균/편차 계산에서만 제외."""
        valid = self._valid_mask_for_columns([x_col, y_col, theta_col])
        x_data = self.df.loc[valid, x_col].astype(float).values
        y_data = self.df.loc[valid, y_col].astype(float).values
        theta_data = self.df.loc[valid, theta_col].astype(float).values

        # 평균 위치
        x_mean = np.mean(x_data)
        y_mean = np.mean(y_data)
        theta_mean = self.circular_mean(theta_data)

        # 표준편차
        sigma_x = np.std(x_data, ddof=1)
        sigma_y = np.std(y_data, ddof=1)

        # Angular error (shortest distance)
        theta_errors = self.angular_diff(theta_data, theta_mean)
        sigma_theta = np.std(theta_errors, ddof=1)

        # 2D position repeatability: combined standard deviation
        sigma_2d = np.sqrt(sigma_x**2 + sigma_y**2)

        # per-point distance from mean (for histogram / ISO 9283)
        position_errors = np.sqrt((x_data - x_mean) ** 2 + (y_data - y_mean) ** 2)

        # ISO 9283 방식: Rp = mean + 3*sigma
        # rp_2d = np.mean(position_errors) + 3 * sigma_2d

        return {
            "mean_x": x_mean,
            "mean_y": y_mean,
            "mean_theta": theta_mean,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "sigma_theta": sigma_theta,
            "sigma_2d": sigma_2d,
            #'Rp_ISO9283': rp_2d,
            "position_errors": position_errors,
            "theta_errors": theta_errors,
        }

    def _empty_trajectory_result(
        self,
        sampling_interval_mm: float,
        x_min: float = 0.0,
        x_max: float = 0.0,
    ) -> Dict:
        """빈 궤적 분석 결과 반환 (데이터 부족 시)"""
        return {
            "sampling_interval": sampling_interval_mm,
            "n_samples": 0,
            "target_x": np.array([]),
            "x_range": (x_min, x_max),
            "sigma_y": np.nan,
            "sigma_theta": np.nan,
            "sigma_y_at_each_x": np.array([]),
            "sigma_theta_at_each_x": np.array([]),
            "y_errors": np.array([]),
            "theta_errors": np.array([]),
            "reference": {"x": np.array([]), "y": np.array([]), "theta": np.array([])},
            "trajectories": [],
        }

    def analyze_trajectory(self, sampling_interval_mm: float = 20.0, camera_prefix: str = "cam_2") -> Dict:
        """궤적 반복정밀도 계산 - X축 기준으로 Y, Yaw 정밀도 분석

        Args:
            sampling_interval_mm: 샘플링 간격 (mm)
            camera_prefix: 카메라 컬럼 접두사 (예: "cam_2", "cam_3")
        """
        n_trials = len(self.df)
        trajectories = []
        for trial_idx in range(n_trials):
            x_vals = []
            y_vals = []
            theta_vals = []

            for wp_idx in range(MAX_TRAJECTORY_POINTS):
                x_col = f"{camera_prefix}_x_{wp_idx}"
                y_col = f"{camera_prefix}_y_{wp_idx}"
                theta_col = f"{camera_prefix}_rz_{wp_idx}"

                if x_col not in self.df.columns:
                    break

                x = self.df.iloc[trial_idx][x_col]
                y = self.df.iloc[trial_idx][y_col]
                theta = self.df.iloc[trial_idx][theta_col]

                if pd.isna(x) or pd.isna(y) or pd.isna(theta):
                    break

                x_vals.append(float(x))
                y_vals.append(float(y))
                theta_vals.append(float(theta))

            if len(x_vals) > 1:
                x_arr = np.array(x_vals)
                y_arr = np.array(y_vals)
                theta_arr = np.array(theta_vals)

                # np.interp requires monotonically increasing x
                if x_arr[-1] < x_arr[0]:
                    x_arr = x_arr[::-1]
                    y_arr = y_arr[::-1]
                    theta_arr = theta_arr[::-1]

                trajectories.append(
                    {
                        "x": x_arr,
                        "y": y_arr,
                        "theta": theta_arr,
                    }
                )

        print(f"총 {len(trajectories)}개 회차 궤적 로드됨")

        # 궤적이 없으면 빈 결과 반환
        if len(trajectories) == 0:
            print("⚠ 경고: 유효한 궤적 데이터가 없습니다.")
            return self._empty_trajectory_result(sampling_interval_mm)

        # Step 1: Reference trajectory 생성 (X 기준)
        # X의 공통 범위 찾기
        # 궤적 방향에 따라 x가 증가 또는 감소할 수 있음
        x_mins = [traj["x"].min() for traj in trajectories]
        x_maxs = [traj["x"].max() for traj in trajectories]
        
        # 공통 범위: 모든 궤적이 겹치는 구간
        x_common_start = max(x_mins)  # 가장 늦게 시작하는 지점
        x_common_end = min(x_maxs)    # 가장 먼저 끝나는 지점
        
        # x_min_all, x_max_all은 항상 x_min < x_max가 되도록 정렬
        x_min_all = min(x_common_start, x_common_end)
        x_max_all = max(x_common_start, x_common_end)

        print(f"X 공통 범위: {x_min_all:.2f} ~ {x_max_all:.2f} mm")

        # X 범위 유효성 검사 (범위가 너무 작으면 경고)
        x_range = x_max_all - x_min_all
        if x_range < sampling_interval_mm:
            print(f"⚠ 경고: 궤적들의 X 공통 범위({x_range:.2f}mm)가 샘플링 간격({sampling_interval_mm}mm)보다 작습니다")
            return self._empty_trajectory_result(sampling_interval_mm, x_min_all, x_max_all)

        # Step 2: Uniform X sampling positions
        x_range = x_max_all - x_min_all
        n_samples = max(2, int(x_range / sampling_interval_mm) + 1)  # 최소 2개 샘플
        target_x = np.linspace(x_min_all, x_max_all, n_samples)

        print(f"X 샘플링 간격: {sampling_interval_mm} mm, 샘플 개수: {n_samples}")

        # Step 3: 각 X 위치에서 Y와 Theta의 reference 값 계산
        ref_y_at_x = []
        ref_theta_at_x = []

        for x_sample in target_x:
            y_values = []
            theta_values = []

            for traj in trajectories:
                # 이 trial에서 x_sample에 가장 가까운 점 찾기 (interpolation)
                y_interp = np.interp(x_sample, traj["x"], traj["y"])

                # Theta는 unwrap 후 interpolation
                theta_unwrap = np.unwrap(np.deg2rad(traj["theta"]))
                theta_interp = np.interp(x_sample, traj["x"], theta_unwrap)
                theta_interp = np.rad2deg(theta_interp) % 360

                y_values.append(y_interp)
                theta_values.append(theta_interp)

            ref_y_at_x.append(np.mean(y_values))
            ref_theta_at_x.append(self.circular_mean(np.array(theta_values)))

        ref_y_at_x = np.array(ref_y_at_x)
        ref_theta_at_x = np.array(ref_theta_at_x)

        print(f"Reference Y 범위: {ref_y_at_x.min():.2f} ~ {ref_y_at_x.max():.2f} mm")
        print(
            f"Reference Theta 범위: {ref_theta_at_x.min():.2f} ~ {ref_theta_at_x.max():.2f}°"
        )

        # Step 4: 각 trial의 Y, Theta 오차 계산
        y_errors = []  # [n_trials, n_samples]
        theta_errors = []

        for trial_idx, traj in enumerate(trajectories):
            trial_y_at_x = []
            trial_theta_at_x = []

            for x_sample in target_x:
                # Y interpolation
                y_interp = np.interp(x_sample, traj["x"], traj["y"])
                trial_y_at_x.append(y_interp)

                # Theta interpolation
                theta_unwrap = np.unwrap(np.deg2rad(traj["theta"]))
                theta_interp = np.interp(x_sample, traj["x"], theta_unwrap)
                theta_interp = np.rad2deg(theta_interp) % 360
                trial_theta_at_x.append(theta_interp)

            trial_y_at_x = np.array(trial_y_at_x)
            trial_theta_at_x = np.array(trial_theta_at_x)

            # Y error (lateral deviation)
            y_err = trial_y_at_x - ref_y_at_x

            # Angular error
            ang_err = self.angular_diff(trial_theta_at_x, ref_theta_at_x)

            y_errors.append(y_err)
            theta_errors.append(ang_err)

        y_errors = np.array(y_errors)  # [n_trials, n_samples]
        theta_errors = np.array(theta_errors)

        # Step 5: 통계 계산
        # Overall statistics
        sigma_y = np.std(y_errors, ddof=1)
        sigma_theta = np.std(theta_errors, ddof=1)

        # Position-wise statistics
        sigma_y_at_each_x = np.std(y_errors, axis=0, ddof=1)
        sigma_theta_at_each_x = np.std(theta_errors, axis=0, ddof=1)

        return {
            "sampling_interval": sampling_interval_mm,
            "n_samples": n_samples,
            "target_x": target_x,
            "x_range": (x_min_all, x_max_all),
            "sigma_y": sigma_y,
            "sigma_theta": sigma_theta,
            "sigma_y_at_each_x": sigma_y_at_each_x,
            "sigma_theta_at_each_x": sigma_theta_at_each_x,
            "y_errors": y_errors,
            "theta_errors": theta_errors,
            "reference": {"x": target_x, "y": ref_y_at_x, "theta": ref_theta_at_x},
            "trajectories": trajectories,
        }

    def run_analysis(self, sampling_interval_mm: float = 20.0):
        """전체 분석 실행"""
        print("=" * 60)
        print("반복 위치 정밀도 분석 시작")
        print("=" * 60)

        # Cam1 분석
        print("\n[Cam1 - 정지 위치 정밀도]")
        self.results["cam1"] = self.analyze_static_position(
            "cam1", "cam_1_x(mm)", "cam_1_y(mm)", "cam_1_rz(deg)"
        )
        # self._print_static_results('Cam1', self.results['cam1'])

        # Cam2 궤적 분석
        print("\n[Cam2 - 궤적 반복 정밀도]")
        self.results["cam2"] = self.analyze_trajectory(sampling_interval_mm, camera_prefix="cam_2")

        # Cam3 궤적 분석
        print("\n[Cam3 - 궤적 반복 정밀도]")
        self.results["cam3"] = self.analyze_trajectory(sampling_interval_mm, camera_prefix="cam_3")
        # self._print_trajectory_results('Cam2', self.results['cam2'])

    def save_results_to_csv(self, output_dir: str = "outputs"):
        """분석 결과를 CSV 파일로 저장"""

        # 1. Summary CSV (전체 통계)
        summary_data = []

        # Cam1 데이터
        cam1 = self.results["cam1"]
        summary_data.append(
            {
                "Camera": "Cam1",
                "Type": "Static",
                "Mean_X(mm)": cam1["mean_x"],
                "Mean_Y(mm)": cam1["mean_y"],
                "Mean_Theta(deg)": cam1["mean_theta"],
                "Sigma_X(mm)": cam1["sigma_x"],
                "Sigma_Y(mm)": cam1["sigma_y"],
                "Sigma_Theta(deg)": cam1["sigma_theta"],
                "Sigma_2D(mm)": cam1["sigma_2d"],
                #'Rp_ISO9283(mm)': cam1['Rp_ISO9283'],
            }
        )

        # Cam2 전체 통계
        cam2 = self.results["cam2"]
        summary_data.append(
            {
                "Camera": "Cam2",
                "Type": "Trajectory",
                "Mean_X(mm)": np.nan,  # X는 제어 변수
                "Mean_Y(mm)": np.mean(cam2["reference"]["y"]) if cam2.get("reference") else np.nan,
                "Mean_Theta(deg)": np.mean(cam2["reference"]["theta"]) if cam2.get("reference") else np.nan,
                "Sigma_X(mm)": np.nan,  # X는 측정 안함
                "Sigma_Y(mm)": cam2.get("sigma_y", np.nan),
                "Sigma_Theta(deg)": cam2.get("sigma_theta", np.nan),
                "Sigma_2D(mm)": np.nan,  # 궤적은 2D 개념 없음
                #'Rp_ISO9283(mm)': np.nan,
            }
        )

        # Cam3 전체 통계 (Trajectory)
        cam3 = self.results["cam3"]
        summary_data.append(
            {
                "Camera": "Cam3",
                "Type": "Trajectory",
                "Mean_X(mm)": np.nan,  # X는 제어 변수
                "Mean_Y(mm)": np.mean(cam3["reference"]["y"]) if cam3.get("reference") else np.nan,
                "Mean_Theta(deg)": np.mean(cam3["reference"]["theta"]) if cam3.get("reference") else np.nan,
                "Sigma_X(mm)": np.nan,  # X는 측정 안함
                "Sigma_Y(mm)": cam3.get("sigma_y", np.nan),
                "Sigma_Theta(deg)": cam3.get("sigma_theta", np.nan),
                "Sigma_2D(mm)": np.nan,  # 궤적은 2D 개념 없음
                #'Rp_ISO9283(mm)': np.nan,
            }
        )

        summary_df = pd.DataFrame(summary_data)
        summary_path = f"{output_dir}/auto_repeatability_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n[Summary CSV 저장] {summary_path}")

        # 2. Cam2 Detailed CSV (샘플 포인트별)
        cam2_detailed_path = f"{output_dir}/auto_cam2_trajectory_detailed.csv"
        if cam2.get("target_x") is not None and len(cam2["target_x"]) > 0:
            cam2_detailed_data = []
            for i, x_pos in enumerate(cam2["target_x"]):
                cam2_detailed_data.append(
                    {
                        "Sample_Index": i,
                        "X_Position(mm)": x_pos,
                        "Reference_Y(mm)": cam2["reference"]["y"][i],
                        "Reference_Theta(deg)": cam2["reference"]["theta"][i],
                        "Sigma_Y(mm)": cam2["sigma_y_at_each_x"][i],
                        "Sigma_Theta(deg)": cam2["sigma_theta_at_each_x"][i],
                    }
                )
            cam2_detailed_df = pd.DataFrame(cam2_detailed_data)
            cam2_detailed_df.to_csv(cam2_detailed_path, index=False)
            print(f"[Cam2 Detailed CSV 저장] {cam2_detailed_path}")
        else:
            print(f"[Cam2 Detailed CSV 건너뜀] 유효한 trajectory 데이터 없음")

        # 3. Cam3 Detailed CSV (샘플 포인트별, Trajectory)
        cam3_detailed_path = f"{output_dir}/auto_cam3_trajectory_detailed.csv"
        if cam3.get("target_x") is not None and len(cam3["target_x"]) > 0:
            cam3_detailed_data = []
            for i, x_pos in enumerate(cam3["target_x"]):
                cam3_detailed_data.append(
                    {
                        "Sample_Index": i,
                        "X_Position(mm)": x_pos,
                        "Reference_Y(mm)": cam3["reference"]["y"][i],
                        "Reference_Theta(deg)": cam3["reference"]["theta"][i],
                        "Sigma_Y(mm)": cam3["sigma_y_at_each_x"][i],
                        "Sigma_Theta(deg)": cam3["sigma_theta_at_each_x"][i],
                    }
                )
            cam3_detailed_df = pd.DataFrame(cam3_detailed_data)
            cam3_detailed_df.to_csv(cam3_detailed_path, index=False)
            print(f"[Cam3 Detailed CSV 저장] {cam3_detailed_path}")
        else:
            print(f"[Cam3 Detailed CSV 건너뜀] 유효한 trajectory 데이터 없음")

        # 4. Cam1 개별 측정값 (전체 행 기록 유지, 무효 행은 Position_Error/Theta_Error만 NaN)
        valid_cam1 = self._valid_mask_for_columns(["cam_1_x(mm)", "cam_1_y(mm)", "cam_1_rz(deg)"])
        pos_err_cam1 = np.full(len(self.df), np.nan, dtype=float)
        pos_err_cam1[valid_cam1] = cam1["position_errors"]
        theta_err_cam1 = np.full(len(self.df), np.nan, dtype=float)
        theta_err_cam1[valid_cam1] = cam1["theta_errors"]
        cam1_measurements = pd.DataFrame(
            {
                "Trial": range(len(self.df)),
                "X(mm)": self.df["cam_1_x(mm)"],
                "Y(mm)": self.df["cam_1_y(mm)"],
                "Theta(deg)": self.df["cam_1_rz(deg)"],
                "Position_Error(mm)": pos_err_cam1,
                "Theta_Error(deg)": theta_err_cam1,
            }
        )
        cam1_measurements_path = f"{output_dir}/auto_cam1_measurements.csv"
        cam1_measurements.to_csv(cam1_measurements_path, index=False)
        print(f"[Cam1 Measurements CSV 저장] {cam1_measurements_path}")

        return {
            "summary": summary_path,
            "cam2_detailed": cam2_detailed_path,
            "cam3_detailed": cam3_detailed_path,
            "cam1_measurements": cam1_measurements_path,
        }

    def _print_static_results(self, cam_name: str, results: Dict):
        """정지 위치 결과 출력"""
        print(f"  평균 위치: ({results['mean_x']:.3f}, {results['mean_y']:.3f}) mm")
        print(f"  평균 각도: {results['mean_theta']:.3f}°")
        print(f"  σ_x: {results['sigma_x']:.4f} mm")
        print(f"  σ_y: {results['sigma_y']:.4f} mm")
        print(f"  σ_θ: {results['sigma_theta']:.4f}°")
        print(f"  σ_2D (position): {results['sigma_2d']:.4f} mm")
        # print(f"  Rp (ISO 9283): {results['Rp_ISO9283']:.4f} mm")

    def _print_trajectory_results(self, cam_name: str, results: Dict):
        """궤적 결과 출력"""
        print(f"  X 범위: {results['x_range'][0]:.2f} ~ {results['x_range'][1]:.2f} mm")
        print(f"  샘플링 간격: {results['sampling_interval']:.1f} mm")
        print(f"  샘플 개수: {results['n_samples']}")
        print(f"  σ_y (lateral deviation): {results['sigma_y']:.4f} mm")
        print(f"  σ_θ (angular deviation): {results['sigma_theta']:.4f}°")
        print(f"  최대 Y 표준편차: {np.max(results['sigma_y_at_each_x']):.4f} mm")
        print(f"  최소 Y 표준편차: {np.min(results['sigma_y_at_each_x']):.4f} mm")

    @staticmethod
    def _zoom_to_data(ax, x_data: np.ndarray, y_data: np.ndarray, margin_ratio: float = 0.15,
                      zoom_x: bool = True, zoom_y: bool = True):
        """축 범위를 데이터의 min-max에 맞게 zoom-in (마진 포함)."""
        if len(x_data) == 0 or len(y_data) == 0:
            return
        if zoom_x:
            x_min, x_max = float(np.min(x_data)), float(np.max(x_data))
            x_margin = max((x_max - x_min) * margin_ratio, 0.5)
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
        if zoom_y:
            y_min, y_max = float(np.min(y_data)), float(np.max(y_data))
            y_margin = max((y_max - y_min) * margin_ratio, 0.5)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)

    def _plot_trajectory_overlay(self, ax, cam_data, cam_name: str):
        """궤적 오버레이 플롯 헬퍼 (Cam2, Cam3 공용)"""
        all_x, all_y = [], []
        if cam_data.get("trajectories") and len(cam_data["trajectories"]) > 0:
            for traj in cam_data["trajectories"]:
                ax.plot(traj["x"], traj["y"], "b-", alpha=0.3, linewidth=0.5)
                all_x.extend(traj["x"])
                all_y.extend(traj["y"])
            if cam_data.get("reference") and len(cam_data["reference"]["x"]) > 0:
                ax.plot(
                    cam_data["reference"]["x"],
                    cam_data["reference"]["y"],
                    "r-",
                    linewidth=2,
                    label="Reference",
                )
            sigma_y_str = f"{cam_data['sigma_y']:.4f}" if not np.isnan(cam_data.get('sigma_y', np.nan)) else "N/A"
            ax.set_title(f"{cam_name} Trajectories\nσ_y={sigma_y_str} mm")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No trajectory data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{cam_name} Trajectories\n(No data)")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.grid(True, alpha=0.3)
        if all_x and all_y:
            self._zoom_to_data(ax, np.array(all_x), np.array(all_y), zoom_x=False, zoom_y=True)

    def _plot_trajectory_sigma_theta(self, ax, cam_data, cam_name: str):
        """궤적 각도 편차 플롯 헬퍼 (Cam2, Cam3 공용)"""
        if cam_data.get("target_x") is not None and len(cam_data["target_x"]) > 0:
            ax.plot(
                cam_data["target_x"], cam_data["sigma_theta_at_each_x"], "g-", linewidth=2
            )
            ax.fill_between(
                cam_data["target_x"],
                0,
                cam_data["sigma_theta_at_each_x"],
                alpha=0.3,
                color="green",
            )
            sigma_theta_str = f"{cam_data['sigma_theta']:.4f}" if not np.isnan(cam_data.get('sigma_theta', np.nan)) else "N/A"
            ax.set_title(f"{cam_name} Yaw Repeatability\nσ_θ={sigma_theta_str}°")
        else:
            ax.text(0.5, 0.5, "No trajectory data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{cam_name} Yaw Repeatability\n(No data)")
        ax.set_xlabel("X Position (mm)")
        ax.set_ylabel("σ_θ (deg)")
        ax.grid(True, alpha=0.3)

    def _plot_trajectory_detailed(self, fig_or_axes, cam_data, cam_name: str):
        """궤적 상세 분석 플롯 헬퍼 (Y/Theta deviation, Cam2/Cam3 공용)"""
        if isinstance(fig_or_axes, tuple):
            ax_y, ax_theta = fig_or_axes
        else:
            ax_y = fig_or_axes.add_subplot(1, 2, 1)
            ax_theta = fig_or_axes.add_subplot(1, 2, 2)

        # Y error along X trajectory
        if cam_data.get("target_x") is not None and len(cam_data["target_x"]) > 0:
            ax_y.plot(
                cam_data["target_x"], cam_data["sigma_y_at_each_x"], "b-", linewidth=2
            )
            ax_y.fill_between(
                cam_data["target_x"],
                0,
                cam_data["sigma_y_at_each_x"],
                alpha=0.3,
                color="blue",
            )
            sigma_y_str = f'{cam_data["sigma_y"]:.4f}' if not np.isnan(cam_data.get("sigma_y", np.nan)) else "N/A"
            ax_y.set_title(f'{cam_name} Y Repeatability (Lateral Deviation)\nOverall σ_y={sigma_y_str} mm')
        else:
            ax_y.text(0.5, 0.5, "No trajectory data", ha='center', va='center', transform=ax_y.transAxes)
            ax_y.set_title(f"{cam_name} Y Repeatability\n(No data)")
        ax_y.set_xlabel("X Position (mm)")
        ax_y.set_ylabel("σ_y (mm)")
        ax_y.grid(True, alpha=0.3)

        # Theta error along X trajectory
        if cam_data.get("target_x") is not None and len(cam_data["target_x"]) > 0:
            ax_theta.plot(
                cam_data["target_x"], cam_data["sigma_theta_at_each_x"], "g-", linewidth=2
            )
            ax_theta.fill_between(
                cam_data["target_x"],
                0,
                cam_data["sigma_theta_at_each_x"],
                alpha=0.3,
                color="green",
            )
            sigma_theta_str = f'{cam_data["sigma_theta"]:.4f}' if not np.isnan(cam_data.get("sigma_theta", np.nan)) else "N/A"
            ax_theta.set_title(f'{cam_name} Yaw Repeatability (Angular Deviation)\nOverall σ_θ={sigma_theta_str}°')
        else:
            ax_theta.text(0.5, 0.5, "No trajectory data", ha='center', va='center', transform=ax_theta.transAxes)
            ax_theta.set_title(f"{cam_name} Yaw Repeatability\n(No data)")
        ax_theta.set_xlabel("X Position (mm)")
        ax_theta.set_ylabel("σ_θ (deg)")
        ax_theta.grid(True, alpha=0.3)

        return ax_y, ax_theta

    def plot_results(self, output_dir: str = "outputs"):
        """결과 시각화 — 단일 PNG (3x3 layout)

        Row 1: Cam1 Position, Cam1 Yaw, Cam1 X/Y Trend
        Row 2: Cam2 Trajectory, Cam2 Y σ, Cam2 Yaw σ
        Row 3: Cam3 Trajectory, Cam3 Y σ, Cam3 Yaw σ
        """
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))

        cam1_data = self.results["cam1"]
        cam2_data = self.results["cam2"]
        cam3_data = self.results["cam3"]

        valid_cam1 = self._valid_mask_for_columns(["cam_1_x(mm)", "cam_1_y(mm)", "cam_1_rz(deg)"])

        # ── Row 1: Cam1 ──
        ax = axes[0, 0]
        x_data = self.df.loc[valid_cam1, "cam_1_x(mm)"].astype(float).values
        y_data = self.df.loc[valid_cam1, "cam_1_y(mm)"].astype(float).values
        ax.scatter(x_data, y_data, alpha=0.6, s=50, c="blue")
        ax.plot(cam1_data["mean_x"], cam1_data["mean_y"], "r*", markersize=15, label="Mean")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(f"Cam1 Position\nσ_2D={cam1_data['sigma_2d']:.4f} mm")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._zoom_to_data(ax, x_data, y_data)

        ax = axes[0, 1]
        theta_data_cam1 = self.df.loc[valid_cam1, "cam_1_rz(deg)"].astype(float).values
        trials = np.arange(1, len(theta_data_cam1) + 1)
        ax.scatter(trials, theta_data_cam1, alpha=0.6, s=50, c="green")
        ax.axhline(
            y=cam1_data["mean_theta"], color="r", linestyle="--", linewidth=2,
            label=f"Mean={cam1_data['mean_theta']:.2f}°",
        )
        ax.fill_between(
            trials,
            cam1_data["mean_theta"] - cam1_data["sigma_theta"],
            cam1_data["mean_theta"] + cam1_data["sigma_theta"],
            alpha=0.2, color="red",
            label=f"±σ={cam1_data['sigma_theta']:.4f}°",
        )
        ax.set_xlabel("Trial")
        ax.set_ylabel("Yaw (deg)")
        ax.set_title(f"Cam1 Yaw Repeatability\nσ_θ={cam1_data['sigma_theta']:.4f}°")
        ax.legend(loc="upper right", fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)

        ax = axes[0, 2]
        ax.plot(trials, x_data, "o-", alpha=0.7, markersize=4, color="blue", label="X")
        ax.plot(trials, y_data, "s-", alpha=0.7, markersize=4, color="green", label="Y")
        ax.axhline(y=cam1_data["mean_x"], color="blue", linestyle="--", alpha=0.4)
        ax.axhline(y=cam1_data["mean_y"], color="green", linestyle="--", alpha=0.4)
        ax.set_xlabel("Trial")
        ax.set_ylabel("Position (mm)")
        ax.set_title("Cam1 X/Y Trend")
        ax.legend(loc="best", fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)

        # ── Row 2: Cam2 ──
        self._plot_trajectory_overlay(axes[1, 0], cam2_data, "Cam2")
        self._plot_trajectory_detailed((axes[1, 1], axes[1, 2]), cam2_data, "Cam2")

        # ── Row 3: Cam3 ──
        self._plot_trajectory_overlay(axes[2, 0], cam3_data, "Cam3")
        self._plot_trajectory_detailed((axes[2, 1], axes[2, 2]), cam3_data, "Cam3")

        plt.tight_layout()
        fig_path = f"{output_dir}/auto_repeatability_analysis.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"\n[그래프 저장] {fig_path}")

        return fig


def _circular_mean_deg(angles_deg: np.ndarray) -> float:
    """각도의 circular mean (degree)."""
    angles_rad = np.deg2rad(angles_deg)
    return np.rad2deg(np.arctan2(np.mean(np.sin(angles_rad)), np.mean(np.cos(angles_rad))))


def _angular_diff_deg(angles_deg: np.ndarray, ref_deg: float) -> np.ndarray:
    """각도와 기준 각도의 shortest angular distance (degree)."""
    diff = angles_deg - ref_deg
    return (diff + 180) % 360 - 180


class ManualRepeatability:
    """Cam1 수동 측정 반복정밀도 분석 (x, y, rz CSV). cam1_measurements.csv와 동일 형식으로 저장."""

    # 수동 측정 raw CSV 형식 (cam_1_x(mm), cam_1_y(mm), cam_1_rz(deg)) → x, y, rz(rad) 매핑
    COLUMN_ALIASES = [
        (["x", "y", "rz"], None),  # rz 단위: rad
        (["cam_1_x(mm)", "cam_1_y(mm)", "cam_1_rz(deg)"], "deg"),  # rz 단위: deg → rad 변환
    ]

    def __init__(self, csv_path: str):
        """
        CSV 파일을 로드합니다. 필수 컬럼: x, y, rz (rz 단위: rad)
        또는 cam_1_x(mm), cam_1_y(mm), cam_1_rz(deg) (rz 단위: deg).

        Args:
            csv_path: CSV 파일 경로
        """
        self.df = pd.read_csv(csv_path)
        required = ["x", "y", "rz"]
        if all(c in self.df.columns for c in required):
            self.results = {}
            return
        # 대체 컬럼명 지원: cam_1_x(mm), cam_1_y(mm), cam_1_rz(deg) 등
        for cols, rz_unit in self.COLUMN_ALIASES:
            if cols == ["x", "y", "rz"]:
                continue
            if all(c in self.df.columns for c in cols):
                self.df = self.df.rename(columns={cols[0]: "x", cols[1]: "y", cols[2]: "rz"})
                if rz_unit == "deg":
                    self.df["rz"] = np.deg2rad(self.df["rz"])
                self._rz_unit = rz_unit or "rad"
                self.results = {}
                return
        missing = [c for c in required if c not in self.df.columns]
        raise ValueError(f"CSV에 필수 컬럼이 없습니다: {missing}. 지원 형식: x,y,rz(rad) 또는 cam_1_x(mm),cam_1_y(mm),cam_1_rz(deg)")

    @staticmethod
    def _zoom_to_data_static(ax, x_data: np.ndarray, y_data: np.ndarray, margin_ratio: float = 0.15):
        """축 범위를 데이터의 min-max에 맞게 zoom-in (마진 포함)."""
        if len(x_data) == 0 or len(y_data) == 0:
            return
        x_min, x_max = float(np.min(x_data)), float(np.max(x_data))
        y_min, y_max = float(np.min(y_data)), float(np.max(y_data))
        x_margin = max((x_max - x_min) * margin_ratio, 0.5)
        y_margin = max((y_max - y_min) * margin_ratio, 0.5)
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

    def _valid_mask_xy_rz(self) -> np.ndarray:
        """x, y, rz 중 NaN·0.0이 아닌 행만 True. (행은 삭제하지 않고, 평균/편차 계산 시에만 사용.)"""
        valid = self.df["x"].notna() & self.df["y"].notna() & self.df["rz"].notna()
        xv = pd.to_numeric(self.df["x"], errors="coerce")
        yv = pd.to_numeric(self.df["y"], errors="coerce")
        zv = pd.to_numeric(self.df["rz"], errors="coerce")
        valid &= (xv != 0.0) & (yv != 0.0) & (zv != 0.0)
        return valid.values

    def run_analysis(self) -> None:
        """통계 계산 및 cam1_measurements 형식용 Position_Error, Theta_Error 계산. NaN·0.0 행은 기록은 유지하고 평균/편차 계산에서만 제외."""
        valid = self._valid_mask_xy_rz()
        x = self.df.loc[valid, "x"].astype(float).values
        y = self.df.loc[valid, "y"].astype(float).values
        rz = self.df.loc[valid, "rz"].astype(float).values
        n = len(x)
        if n < 2:
            raise ValueError("최소 2개 이상의 측정값이 필요합니다. (NaN·0.0 제외 후 유효 행이 2개 미만입니다.)")

        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        theta_deg = np.rad2deg(rz)
        mean_theta_deg = _circular_mean_deg(theta_deg)
        position_errors = np.sqrt((x - x_mean) ** 2 + (y - y_mean) ** 2)
        theta_errors = _angular_diff_deg(theta_deg, mean_theta_deg)

        self.results = {
            "n_measurements": n,
            "x_mean": x_mean,
            "x_std": float(np.std(x, ddof=1)),
            "x_min": float(np.min(x)),
            "x_max": float(np.max(x)),
            "y_mean": y_mean,
            "y_std": float(np.std(y, ddof=1)),
            "y_min": float(np.min(y)),
            "y_max": float(np.max(y)),
            "rz_mean": float(np.mean(rz)),
            "rz_std": float(np.std(rz, ddof=1)),
            "rz_min": float(np.min(rz)),
            "rz_max": float(np.max(rz)),
        }
        self.results["x_range"] = self.results["x_max"] - self.results["x_min"]
        self.results["y_range"] = self.results["y_max"] - self.results["y_min"]
        self.results["rz_range"] = self.results["rz_max"] - self.results["rz_min"]
        self.results["x_repeatability"] = 3 * self.results["x_std"]
        self.results["y_repeatability"] = 3 * self.results["y_std"]
        self.results["rz_repeatability"] = 3 * self.results["rz_std"]
        self.results["x_values"] = x
        self.results["y_values"] = y
        self.results["rz_values"] = rz
        self.results["theta_deg"] = theta_deg
        self.results["position_errors"] = position_errors
        self.results["theta_errors"] = theta_errors

    def save_results_to_csv(self, output_dir: str) -> str:
        """cam1_measurements.csv와 동일 컬럼으로 저장. 전체 행 기록 유지, 무효 행은 Position_Error/Theta_Error만 NaN."""
        r = self.results
        n_total = len(self.df)
        valid = self._valid_mask_xy_rz()
        # 유효 행만 계산된 에러 → 전체 길이 배열로 (무효 행은 NaN)
        position_errors_full = np.full(n_total, np.nan, dtype=float)
        position_errors_full[valid] = r["position_errors"]
        theta_errors_full = np.full(n_total, np.nan, dtype=float)
        theta_errors_full[valid] = r["theta_errors"]
        # X, Y, Theta(deg): 원본 df 전체 (rz는 rad → deg 변환)
        theta_deg_full = np.rad2deg(self.df["rz"].astype(float)) if "rz" in self.df.columns else np.full(n_total, np.nan)
        cam1_manual_measurements = pd.DataFrame({
            "Trial": range(n_total),
            "X(mm)": self.df["x"].values,
            "Y(mm)": self.df["y"].values,
            "Theta(deg)": theta_deg_full,
            "Position_Error(mm)": position_errors_full,
            "Theta_Error(deg)": theta_errors_full,
        })
        path = f"{output_dir}/manual_cam1_measurements.csv"
        cam1_manual_measurements.to_csv(path, index=False)
        print(f"[Cam1] Manual measurements CSV 저장: {path}")
        return path

    def plot_results(self, output_dir: str) -> str:
        """반복정밀도 그래프 저장 (Position + Yaw). 반환: 저장된 PNG 경로."""
        r = self.results
        n = r["n_measurements"]
        x_v, y_v = r["x_values"], r["y_values"]
        theta_deg = r["theta_deg"]
        x_mean, y_mean = r["x_mean"], r["y_mean"]
        x_std, y_std = r["x_std"], r["y_std"]
        mean_theta_deg = _circular_mean_deg(theta_deg)
        sigma_theta = np.std(_angular_diff_deg(theta_deg, mean_theta_deg), ddof=1)
        sigma_2d = np.sqrt(x_std**2 + y_std**2)

        fig = plt.figure(figsize=(12, 5))

        # 1) Position scatter
        ax1 = plt.subplot(1, 2, 1)
        ax1.scatter(x_v, y_v, alpha=0.6, s=50, c="blue")
        ax1.plot(x_mean, y_mean, "r*", markersize=15, label="Mean")
        ax1.axhline(y_mean, color="gray", linestyle="--", alpha=0.5)
        ax1.axvline(x_mean, color="gray", linestyle="--", alpha=0.5)
        ax1.set_xlabel("X (mm)")
        ax1.set_ylabel("Y (mm)")
        ax1.set_title(f"Cam1 Manual Position\n(n={n}, σ_2D={sigma_2d:.4f} mm)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        self._zoom_to_data_static(ax1, x_v, y_v)

        # 2) Yaw (θ) repeatability
        ax2 = plt.subplot(1, 2, 2)
        trials = np.arange(1, len(theta_deg) + 1)
        ax2.scatter(trials, theta_deg, alpha=0.6, s=50, c="green")
        ax2.axhline(mean_theta_deg, color="r", linestyle="--", linewidth=2, label=f"Mean={mean_theta_deg:.2f}°")
        ax2.fill_between(
            trials,
            mean_theta_deg - sigma_theta,
            mean_theta_deg + sigma_theta,
            alpha=0.2,
            color="red",
            label=f"±σ={sigma_theta:.4f}°",
        )
        ax2.set_xlabel("Trial")
        ax2.set_ylabel("Yaw (deg)")
        ax2.set_title(f"Cam1 Manual Yaw Repeatability\nσ_θ={sigma_theta:.4f}°")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = f"{output_dir}/manual_cam1_repeatability_analysis.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[Cam1] Manual repeatability analysis PNG 저장: {fig_path}")
        return fig_path


if __name__ == "__main__":
    # 분석 실행
    parser = argparse.ArgumentParser(description="Trajectory Repeatability Analysis")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/20251211-143727_zoom1_raw_data.csv",
        help="CSV file path",
    )
    parser.add_argument(
        "--sampling_interval_mm",
        type=float,
        default=50.0,
        help="Sampling interval in mm",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for CSV files",
    )
    args = parser.parse_args()

    analyzer = TrajectoryRepeatability(args.csv_path)
    analyzer.run_analysis(sampling_interval_mm=args.sampling_interval_mm)
    analyzer.plot_results(output_dir=args.output_dir)

    # CSV 결과 저장
    csv_paths = analyzer.save_results_to_csv(args.output_dir)

    print("\n" + "=" * 60)
    print("분석 완료!")
    print("=" * 60)
