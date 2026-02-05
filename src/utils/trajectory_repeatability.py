import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend (no GUI required)
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from typing import Tuple, List, Dict
import argparse


class TrajectoryRepeatability:
    """궤적 및 정지 위치 반복정밀도 분석"""

    # Cam1/Cam3 위치·각도 컬럼 그룹 (이 중 NaN 또는 0.0인 행은 계산에서 제외)
    _POSITION_COLUMN_GROUPS = [
        ["cam_1_x(mm)", "cam_1_y(mm)", "cam_1_rz(deg)"],
        ["cam_3_x(mm)", "cam_3_y(mm)", "cam_3_rz(deg)"],
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
            # 컬럼 수 불일치 오류 발생 시, 헤더 컬럼 수에 맞게 데이터 자르기
            print(f"CSV 파싱 오류 감지: {e}")
            print("헤더 컬럼 수에 맞게 데이터를 자르는 중...")

            # 헤더만 먼저 읽기
            with open(csv_path, "r", encoding="utf-8") as f:
                header_line = f.readline().strip()
            n_cols = len(header_line.split(","))

            # 헤더 컬럼 수만큼만 읽기 (초과 컬럼 무시)
            self.df = pd.read_csv(
                csv_path,
                usecols=range(n_cols),
                on_bad_lines="warn",  # 경고만 출력하고 계속 진행
            )
            print(f"✓ {n_cols}개 컬럼만 로드 완료")

        self.results = {}

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

        # 2D position error
        position_errors = np.sqrt((x_data - x_mean) ** 2 + (y_data - y_mean) ** 2)
        sigma_2d = np.std(position_errors, ddof=1)

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

    def analyze_trajectory(self, sampling_interval_mm: float = 20.0) -> Dict:
        """궤적 반복정밀도 계산 (Cam2) - X축 기준으로 Y, Yaw 정밀도 분석"""
        # Cam2 waypoint 데이터 추출
        n_trials = len(self.df)
        max_waypoints = 100  # 0-99

        trajectories = []
        for trial_idx in range(n_trials):
            x_vals = []
            y_vals = []
            theta_vals = []

            for wp_idx in range(max_waypoints):
                x_col = f"cam_2_x_{wp_idx}"
                y_col = f"cam_2_y_{wp_idx}"
                theta_col = f"cam_2_rz_{wp_idx}"

                if x_col in self.df.columns:
                    x = self.df.iloc[trial_idx][x_col]
                    y = self.df.iloc[trial_idx][y_col]
                    theta = self.df.iloc[trial_idx][theta_col]

                    if pd.notna(x) and pd.notna(y) and pd.notna(theta):
                        x_vals.append(x)
                        y_vals.append(y)
                        theta_vals.append(theta)

            if len(x_vals) > 1:
                trajectories.append(
                    {
                        "x": np.array(x_vals),
                        "y": np.array(y_vals),
                        "theta": np.array(theta_vals),
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

        # Cam3 분석
        print("\n[Cam3 - 정지 위치 정밀도]")
        self.results["cam3"] = self.analyze_static_position(
            "cam3", "cam_3_x(mm)", "cam_3_y(mm)", "cam_3_rz(deg)"
        )
        # self._print_static_results('Cam3', self.results['cam3'])

        # Cam2 궤적 분석
        print("\n[Cam2 - 궤적 반복 정밀도]")
        self.results["cam2"] = self.analyze_trajectory(sampling_interval_mm)
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

        # Cam3 데이터
        cam3 = self.results["cam3"]
        summary_data.append(
            {
                "Camera": "Cam3",
                "Type": "Static",
                "Mean_X(mm)": cam3["mean_x"],
                "Mean_Y(mm)": cam3["mean_y"],
                "Mean_Theta(deg)": cam3["mean_theta"],
                "Sigma_X(mm)": cam3["sigma_x"],
                "Sigma_Y(mm)": cam3["sigma_y"],
                "Sigma_Theta(deg)": cam3["sigma_theta"],
                "Sigma_2D(mm)": cam3["sigma_2d"],
                #'Rp_ISO9283(mm)': cam3['Rp_ISO9283'],
            }
        )

        # Cam2 전체 통계
        cam2 = self.results["cam2"]
        summary_data.append(
            {
                "Camera": "Cam2",
                "Type": "Trajectory",
                "Mean_X(mm)": np.nan,  # X는 제어 변수
                "Mean_Y(mm)": np.mean(cam2["reference"]["y"]),
                "Mean_Theta(deg)": np.mean(cam2["reference"]["theta"]),
                "Sigma_X(mm)": np.nan,  # X는 측정 안함
                "Sigma_Y(mm)": cam2["sigma_y"],
                "Sigma_Theta(deg)": cam2["sigma_theta"],
                "Sigma_2D(mm)": np.nan,  # 궤적은 2D 개념 없음
                #'Rp_ISO9283(mm)': np.nan,
            }
        )

        summary_df = pd.DataFrame(summary_data)
        summary_path = f"{output_dir}/repeatability_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n[Summary CSV 저장] {summary_path}")

        # 2. Cam2 Detailed CSV (샘플 포인트별)
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
        cam2_detailed_path = f"{output_dir}/cam2_trajectory_detailed.csv"
        cam2_detailed_df.to_csv(cam2_detailed_path, index=False)
        print(f"[Cam2 Detailed CSV 저장] {cam2_detailed_path}")

        # 3. Cam1 & Cam3 개별 측정값 (전체 행 기록 유지, 무효 행은 Position_Error/Theta_Error만 NaN)
        valid_cam1 = self._valid_mask_for_columns(["cam_1_x(mm)", "cam_1_y(mm)", "cam_1_rz(deg)"])
        valid_cam3 = self._valid_mask_for_columns(["cam_3_x(mm)", "cam_3_y(mm)", "cam_3_rz(deg)"])
        pos_err_cam1 = np.full(len(self.df), np.nan, dtype=float)
        pos_err_cam1[valid_cam1] = cam1["position_errors"]
        theta_err_cam1 = np.full(len(self.df), np.nan, dtype=float)
        theta_err_cam1[valid_cam1] = cam1["theta_errors"]
        pos_err_cam3 = np.full(len(self.df), np.nan, dtype=float)
        pos_err_cam3[valid_cam3] = cam3["position_errors"]
        theta_err_cam3 = np.full(len(self.df), np.nan, dtype=float)
        theta_err_cam3[valid_cam3] = cam3["theta_errors"]
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
        cam1_measurements_path = f"{output_dir}/cam1_measurements.csv"
        cam1_measurements.to_csv(cam1_measurements_path, index=False)
        print(f"[Cam1 Measurements CSV 저장] {cam1_measurements_path}")

        cam3_measurements = pd.DataFrame(
            {
                "Trial": range(len(self.df)),
                "X(mm)": self.df["cam_3_x(mm)"],
                "Y(mm)": self.df["cam_3_y(mm)"],
                "Theta(deg)": self.df["cam_3_rz(deg)"],
                "Position_Error(mm)": pos_err_cam3,
                "Theta_Error(deg)": theta_err_cam3,
            }
        )
        cam3_measurements_path = f"{output_dir}/cam3_measurements.csv"
        cam3_measurements.to_csv(cam3_measurements_path, index=False)
        print(f"[Cam3 Measurements CSV 저장] {cam3_measurements_path}")

        return {
            "summary": summary_path,
            "cam2_detailed": cam2_detailed_path,
            "cam1_measurements": cam1_measurements_path,
            "cam3_measurements": cam3_measurements_path,
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

    def plot_results(self, output_dir: str = "outputs"):
        """결과 시각화"""
        # Figure 1: Position Repeatability (2x4 layout)
        fig1 = plt.figure(figsize=(20, 10))

        cam1_data = self.results["cam1"]
        cam3_data = self.results["cam3"]
        cam2_data = self.results["cam2"]

        # Helper function for bin calculation
        def get_bins(data, max_bins=20):
            if len(data) < 2:
                return 1
            data_range = np.ptp(data)  # peak-to-peak (max - min)
            if data_range == 0:
                return 1
            return min(max_bins, max(1, len(data) // 5))

        # Row 1: Position Repeatability (스캐터는 통계와 동일하게 유효 행만 표시)
        valid_cam1 = self._valid_mask_for_columns(["cam_1_x(mm)", "cam_1_y(mm)", "cam_1_rz(deg)"])
        valid_cam3 = self._valid_mask_for_columns(["cam_3_x(mm)", "cam_3_y(mm)", "cam_3_rz(deg)"])
        # Cam1 Position scatter plot
        ax1 = plt.subplot(2, 4, 1)
        x_data = self.df.loc[valid_cam1, "cam_1_x(mm)"].astype(float).values
        y_data = self.df.loc[valid_cam1, "cam_1_y(mm)"].astype(float).values
        ax1.scatter(x_data, y_data, alpha=0.6, s=50, c="blue")
        ax1.plot(
            cam1_data["mean_x"], cam1_data["mean_y"], "r*", markersize=15, label="Mean"
        )
        ax1.set_xlabel("X (mm)")
        ax1.set_ylabel("Y (mm)")
        ax1.set_title(f"Cam1 Position\nσ_2D={cam1_data['sigma_2d']:.4f} mm")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis("equal")

        # Cam3 Position scatter plot
        ax2 = plt.subplot(2, 4, 2)
        x_data = self.df.loc[valid_cam3, "cam_3_x(mm)"].astype(float).values
        y_data = self.df.loc[valid_cam3, "cam_3_y(mm)"].astype(float).values
        ax2.scatter(x_data, y_data, alpha=0.6, s=50, c="blue")
        ax2.plot(
            cam3_data["mean_x"], cam3_data["mean_y"], "r*", markersize=15, label="Mean"
        )
        ax2.set_xlabel("X (mm)")
        ax2.set_ylabel("Y (mm)")
        ax2.set_title(f"Cam3 Position\nσ_2D={cam3_data['sigma_2d']:.4f} mm")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis("equal")

        # Position Error Histogram (Cam1 & Cam3)
        ax3 = plt.subplot(2, 4, 3)
        cam1_pos_errors = cam1_data["position_errors"]
        cam3_pos_errors = cam3_data["position_errors"]

        cam1_bins = get_bins(cam1_pos_errors)
        cam3_bins = get_bins(cam3_pos_errors)

        if len(cam1_pos_errors) > 0:
            ax3.hist(
                cam1_pos_errors,
                bins=cam1_bins,
                alpha=0.5,
                label="Cam1",
                edgecolor="black",
                color="blue",
            )
        if len(cam3_pos_errors) > 0:
            ax3.hist(
                cam3_pos_errors,
                bins=cam3_bins,
                alpha=0.5,
                label="Cam3",
                edgecolor="black",
                color="orange",
            )
        ax3.set_xlabel("2D Position Error (mm)")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Cam1 & Cam3 Position Error Distribution")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Cam2 trajectories overlay (X-Y plot)
        ax4 = plt.subplot(2, 4, 4)
        if len(cam2_data["trajectories"]) > 0:
            for traj in cam2_data["trajectories"]:
                ax4.plot(traj["x"], traj["y"], "b-", alpha=0.3, linewidth=0.5)
            if len(cam2_data["reference"]["x"]) > 0:
                ax4.plot(
                    cam2_data["reference"]["x"],
                    cam2_data["reference"]["y"],
                    "r-",
                    linewidth=2,
                    label="Reference",
                )
            sigma_y_str = f"{cam2_data['sigma_y']:.4f}" if not np.isnan(cam2_data['sigma_y']) else "N/A"
            ax4.set_title(f"Cam2 Trajectories\nσ_y={sigma_y_str} mm")
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, "No trajectory data", ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title("Cam2 Trajectories\n(No data)")
        ax4.set_xlabel("X (mm)")
        ax4.set_ylabel("Y (mm)")
        ax4.grid(True, alpha=0.3)
        ax4.axis("equal")

        # Row 2: Yaw (θ) Repeatability
        # Cam1 Yaw scatter plot (Trial vs Theta)
        ax5 = plt.subplot(2, 4, 5)
        theta_data_cam1 = self.df.loc[valid_cam1, "cam_1_rz(deg)"].astype(float).values
        trials = np.arange(len(theta_data_cam1))
        ax5.scatter(trials, theta_data_cam1, alpha=0.6, s=50, c="green")
        ax5.axhline(
            y=cam1_data["mean_theta"],
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Mean={cam1_data['mean_theta']:.2f}°",
        )
        ax5.fill_between(
            trials,
            cam1_data["mean_theta"] - cam1_data["sigma_theta"],
            cam1_data["mean_theta"] + cam1_data["sigma_theta"],
            alpha=0.2,
            color="red",
            label=f"±σ={cam1_data['sigma_theta']:.4f}°",
        )
        ax5.set_xlabel("Trial")
        ax5.set_ylabel("Yaw (deg)")
        ax5.set_title(f"Cam1 Yaw Repeatability\nσ_θ={cam1_data['sigma_theta']:.4f}°")
        ax5.legend(loc="upper right", fontsize=8)
        ax5.grid(True, alpha=0.3)

        # Cam3 Yaw scatter plot (Trial vs Theta)
        ax6 = plt.subplot(2, 4, 6)
        theta_data_cam3 = self.df.loc[valid_cam3, "cam_3_rz(deg)"].astype(float).values
        trials = np.arange(len(theta_data_cam3))
        ax6.scatter(trials, theta_data_cam3, alpha=0.6, s=50, c="green")
        ax6.axhline(
            y=cam3_data["mean_theta"],
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Mean={cam3_data['mean_theta']:.2f}°",
        )
        ax6.fill_between(
            trials,
            cam3_data["mean_theta"] - cam3_data["sigma_theta"],
            cam3_data["mean_theta"] + cam3_data["sigma_theta"],
            alpha=0.2,
            color="red",
            label=f"±σ={cam3_data['sigma_theta']:.4f}°",
        )
        ax6.set_xlabel("Trial")
        ax6.set_ylabel("Yaw (deg)")
        ax6.set_title(f"Cam3 Yaw Repeatability\nσ_θ={cam3_data['sigma_theta']:.4f}°")
        ax6.legend(loc="upper right", fontsize=8)
        ax6.grid(True, alpha=0.3)

        # Yaw Error Histogram (Cam1 & Cam3)
        ax7 = plt.subplot(2, 4, 7)
        cam1_theta_errors = cam1_data["theta_errors"]
        cam3_theta_errors = cam3_data["theta_errors"]

        cam1_theta_bins = get_bins(cam1_theta_errors)
        cam3_theta_bins = get_bins(cam3_theta_errors)

        if len(cam1_theta_errors) > 0:
            ax7.hist(
                cam1_theta_errors,
                bins=cam1_theta_bins,
                alpha=0.5,
                label="Cam1",
                edgecolor="black",
                color="green",
            )
        if len(cam3_theta_errors) > 0:
            ax7.hist(
                cam3_theta_errors,
                bins=cam3_theta_bins,
                alpha=0.5,
                label="Cam3",
                edgecolor="black",
                color="purple",
            )
        ax7.set_xlabel("Yaw Error (deg)")
        ax7.set_ylabel("Frequency")
        ax7.set_title("Cam1 & Cam3 Yaw Error Distribution")
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # Cam2 Angular error along X trajectory
        ax8 = plt.subplot(2, 4, 8)
        if len(cam2_data["target_x"]) > 0:
            ax8.plot(
                cam2_data["target_x"], cam2_data["sigma_theta_at_each_x"], "g-", linewidth=2
            )
            ax8.fill_between(
                cam2_data["target_x"],
                0,
                cam2_data["sigma_theta_at_each_x"],
                alpha=0.3,
                color="green",
            )
            sigma_theta_str = f"{cam2_data['sigma_theta']:.4f}" if not np.isnan(cam2_data['sigma_theta']) else "N/A"
            ax8.set_title(f"Cam2 Yaw Repeatability\nσ_θ={sigma_theta_str}°")
        else:
            ax8.text(0.5, 0.5, "No trajectory data", ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title("Cam2 Yaw Repeatability\n(No data)")
        ax8.set_xlabel("X Position (mm)")
        ax8.set_ylabel("σ_θ (deg)")
        ax8.grid(True, alpha=0.3)

        plt.tight_layout()
        fig1_path = f"{output_dir}/repeatability_analysis.png"
        plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
        print(f"\n[그래프 저장] {fig1_path}")

        # Figure 2: Cam2 Detailed Analysis (Y deviation along trajectory)
        fig2 = plt.figure(figsize=(12, 5))

        # Y error along X trajectory
        ax_y = plt.subplot(1, 2, 1)
        if len(cam2_data["target_x"]) > 0:
            ax_y.plot(
                cam2_data["target_x"], cam2_data["sigma_y_at_each_x"], "b-", linewidth=2
            )
            ax_y.fill_between(
                cam2_data["target_x"],
                0,
                cam2_data["sigma_y_at_each_x"],
                alpha=0.3,
                color="blue",
            )
            sigma_y_str = f'{cam2_data["sigma_y"]:.4f}' if not np.isnan(cam2_data["sigma_y"]) else "N/A"
            ax_y.set_title(f'Cam2 Y Repeatability (Lateral Deviation)\nOverall σ_y={sigma_y_str} mm')
        else:
            ax_y.text(0.5, 0.5, "No trajectory data", ha='center', va='center', transform=ax_y.transAxes)
            ax_y.set_title("Cam2 Y Repeatability\n(No data)")
        ax_y.set_xlabel("X Position (mm)")
        ax_y.set_ylabel("σ_y (mm)")
        ax_y.grid(True, alpha=0.3)

        # Theta error along X trajectory
        ax_theta = plt.subplot(1, 2, 2)
        if len(cam2_data["target_x"]) > 0:
            ax_theta.plot(
                cam2_data["target_x"], cam2_data["sigma_theta_at_each_x"], "g-", linewidth=2
            )
            ax_theta.fill_between(
                cam2_data["target_x"],
                0,
                cam2_data["sigma_theta_at_each_x"],
                alpha=0.3,
                color="green",
            )
            sigma_theta_str = f'{cam2_data["sigma_theta"]:.4f}' if not np.isnan(cam2_data["sigma_theta"]) else "N/A"
            ax_theta.set_title(f'Cam2 Yaw Repeatability (Angular Deviation)\nOverall σ_θ={sigma_theta_str}°')
        else:
            ax_theta.text(0.5, 0.5, "No trajectory data", ha='center', va='center', transform=ax_theta.transAxes)
            ax_theta.set_title("Cam2 Yaw Repeatability\n(No data)")
        ax_theta.set_xlabel("X Position (mm)")
        ax_theta.set_ylabel("σ_θ (deg)")
        ax_theta.grid(True, alpha=0.3)

        plt.tight_layout()
        fig2_path = f"{output_dir}/cam2_trajectory_detailed.png"
        plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
        print(f"[그래프 저장] {fig2_path}")

        return fig1, fig2


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
        """cam1_measurements.csv와 동일 컬럼으로 저장: Trial, X(mm), Y(mm), Theta(deg), Position_Error(mm), Theta_Error(deg)."""
        r = self.results
        n = r["n_measurements"]
        cam1_manual_measurements = pd.DataFrame({
            "Trial": range(n),
            "X(mm)": r["x_values"],
            "Y(mm)": r["y_values"],
            "Theta(deg)": r["theta_deg"],
            "Position_Error(mm)": r["position_errors"],
            "Theta_Error(deg)": r["theta_errors"],
        })
        path = f"{output_dir}/cam1_manual_measurements.csv"
        cam1_manual_measurements.to_csv(path, index=False)
        print(f"[Cam1] Manual measurements CSV 저장: {path}")
        return path

    def plot_results(self, output_dir: str) -> str:
        """반복정밀도 그래프 저장. 반환: 저장된 PNG 경로. (cam1_measurements / repeatability_analysis와 일관된 이름)"""
        r = self.results
        n = r["n_measurements"]
        x_v, y_v = r["x_values"], r["y_values"]
        x_mean, y_mean = r["x_mean"], r["y_mean"]
        x_std, y_std = r["x_std"], r["y_std"]
        x_rep, y_rep = r["x_repeatability"], r["y_repeatability"]

        fig = plt.figure(figsize=(14, 5))

        ax1 = plt.subplot(1, 3, 1)
        ax1.scatter(x_v, y_v, alpha=0.6, s=50, c="blue")
        ax1.plot(x_mean, y_mean, "r*", markersize=15, label="Mean")
        ax1.axhline(y_mean, color="gray", linestyle="--", alpha=0.5)
        ax1.axvline(x_mean, color="gray", linestyle="--", alpha=0.5)
        ax1.set_xlabel("X (mm)")
        ax1.set_ylabel("Y (mm)")
        ax1.set_title(f"Cam1 Manual Position\n(n={n}, 3σ_x={x_rep:.2f}, 3σ_y={y_rep:.2f} mm)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis("equal")

        ax2 = plt.subplot(1, 3, 2)
        ax2.hist(x_v, bins=min(20, max(1, n // 2)), alpha=0.7, color="blue", edgecolor="black")
        ax2.axvline(x_mean, color="r", linestyle="--", linewidth=2, label=f"Mean={x_mean:.2f}")
        ax2.axvline(x_mean - 3 * x_std, color="orange", linestyle=":", alpha=0.8)
        ax2.axvline(x_mean + 3 * x_std, color="orange", linestyle=":", alpha=0.8, label=f"±3σ={x_rep:.2f} mm")
        ax2.set_xlabel("X (mm)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("X Distribution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = plt.subplot(1, 3, 3)
        ax3.hist(y_v, bins=min(20, max(1, n // 2)), alpha=0.7, color="green", edgecolor="black")
        ax3.axvline(y_mean, color="r", linestyle="--", linewidth=2, label=f"Mean={y_mean:.2f}")
        ax3.axvline(y_mean - 3 * y_std, color="orange", linestyle=":", alpha=0.8)
        ax3.axvline(y_mean + 3 * y_std, color="orange", linestyle=":", alpha=0.8, label=f"±3σ={y_rep:.2f} mm")
        ax3.set_xlabel("Y (mm)")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Y Distribution")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = f"{output_dir}/cam1_manual_repeatability_analysis.png"
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
