import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no GUI required)
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from typing import Tuple, List, Dict
import argparse

class TrajectoryRepeatability:
    """궤적 및 정지 위치 반복정밀도 분석"""
    
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.results = {}
        
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
    
    def interpolate_trajectory(self, x: np.ndarray, y: np.ndarray, theta: np.ndarray, 
                               arc_length: np.ndarray, target_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Arc-length 기반으로 궤적을 uniform sampling"""
        # x, y는 linear interpolation
        interp_x = interp1d(arc_length, x, kind='linear', bounds_error=False, fill_value='extrapolate')
        interp_y = interp1d(arc_length, y, kind='linear', bounds_error=False, fill_value='extrapolate')
        
        # theta는 unwrap 후 interpolation, 다시 wrap
        theta_unwrap = np.unwrap(np.deg2rad(theta))
        interp_theta = interp1d(arc_length, theta_unwrap, kind='linear', bounds_error=False, fill_value='extrapolate')
        
        x_resampled = interp_x(target_s)
        y_resampled = interp_y(target_s)
        theta_resampled = np.rad2deg(interp_theta(target_s)) % 360
        
        return x_resampled, y_resampled, theta_resampled
    
    def analyze_static_position(self, cam_name: str, x_col: str, y_col: str, theta_col: str) -> Dict:
        """정지 위치 반복정밀도 계산 (Cam1, Cam3)"""
        x_data = self.df[x_col].values
        y_data = self.df[y_col].values
        theta_data = self.df[theta_col].values
        
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
        position_errors = np.sqrt((x_data - x_mean)**2 + (y_data - y_mean)**2)
        sigma_2d = np.std(position_errors, ddof=1)
        
        # ISO 9283 방식: Rp = mean + 3*sigma
        #rp_2d = np.mean(position_errors) + 3 * sigma_2d
        
        return {
            'mean_x': x_mean,
            'mean_y': y_mean,
            'mean_theta': theta_mean,
            'sigma_x': sigma_x,
            'sigma_y': sigma_y,
            'sigma_theta': sigma_theta,
            'sigma_2d': sigma_2d,
            #'Rp_ISO9283': rp_2d,
            'position_errors': position_errors,
            'theta_errors': theta_errors
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
                x_col = f'cam_2_x_{wp_idx}'
                y_col = f'cam_2_y_{wp_idx}'
                theta_col = f'cam_2_rz_{wp_idx}'
                
                if x_col in self.df.columns:
                    x = self.df.iloc[trial_idx][x_col]
                    y = self.df.iloc[trial_idx][y_col]
                    theta = self.df.iloc[trial_idx][theta_col]
                    
                    if pd.notna(x) and pd.notna(y) and pd.notna(theta):
                        x_vals.append(x)
                        y_vals.append(y)
                        theta_vals.append(theta)
            
            if len(x_vals) > 1:
                trajectories.append({
                    'x': np.array(x_vals),
                    'y': np.array(y_vals),
                    'theta': np.array(theta_vals)
                })
        
        print(f"총 {len(trajectories)}개 회차 궤적 로드됨")
        
        # Step 1: Reference trajectory 생성 (X 기준)
        # X의 공통 범위 찾기
        x_min_all = max([traj['x'].min() for traj in trajectories])
        x_max_all = min([traj['x'].max() for traj in trajectories])
        
        print(f"X 공통 범위: {x_min_all:.2f} ~ {x_max_all:.2f} mm")
        
        # Step 2: Uniform X sampling positions
        n_samples = int((x_max_all - x_min_all) / sampling_interval_mm) + 1
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
                y_interp = np.interp(x_sample, traj['x'], traj['y'])
                
                # Theta는 unwrap 후 interpolation
                theta_unwrap = np.unwrap(np.deg2rad(traj['theta']))
                theta_interp = np.interp(x_sample, traj['x'], theta_unwrap)
                theta_interp = np.rad2deg(theta_interp) % 360
                
                y_values.append(y_interp)
                theta_values.append(theta_interp)
            
            ref_y_at_x.append(np.mean(y_values))
            ref_theta_at_x.append(self.circular_mean(np.array(theta_values)))
        
        ref_y_at_x = np.array(ref_y_at_x)
        ref_theta_at_x = np.array(ref_theta_at_x)
        
        print(f"Reference Y 범위: {ref_y_at_x.min():.2f} ~ {ref_y_at_x.max():.2f} mm")
        print(f"Reference Theta 범위: {ref_theta_at_x.min():.2f} ~ {ref_theta_at_x.max():.2f}°")
        
        # Step 4: 각 trial의 Y, Theta 오차 계산
        y_errors = []  # [n_trials, n_samples]
        theta_errors = []
        
        for trial_idx, traj in enumerate(trajectories):
            trial_y_at_x = []
            trial_theta_at_x = []
            
            for x_sample in target_x:
                # Y interpolation
                y_interp = np.interp(x_sample, traj['x'], traj['y'])
                trial_y_at_x.append(y_interp)
                
                # Theta interpolation
                theta_unwrap = np.unwrap(np.deg2rad(traj['theta']))
                theta_interp = np.interp(x_sample, traj['x'], theta_unwrap)
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
            'sampling_interval': sampling_interval_mm,
            'n_samples': n_samples,
            'target_x': target_x,
            'x_range': (x_min_all, x_max_all),
            'sigma_y': sigma_y,
            'sigma_theta': sigma_theta,
            'sigma_y_at_each_x': sigma_y_at_each_x,
            'sigma_theta_at_each_x': sigma_theta_at_each_x,
            'y_errors': y_errors,
            'theta_errors': theta_errors,
            'reference': {
                'x': target_x,
                'y': ref_y_at_x,
                'theta': ref_theta_at_x
            },
            'trajectories': trajectories
        }
    
    def run_analysis(self, sampling_interval_mm: float = 20.0):
        """전체 분석 실행"""
        print("=" * 60)
        print("반복 위치 정밀도 분석 시작")
        print("=" * 60)
        
        # Cam1 분석
        print("\n[Cam1 - 정지 위치 정밀도]")
        self.results['cam1'] = self.analyze_static_position(
            'cam1', 'cam_1_x(mm)', 'cam_1_y(mm)', 'cam_1_rz(deg)'
        )
        #self._print_static_results('Cam1', self.results['cam1'])
        
        # Cam3 분석
        print("\n[Cam3 - 정지 위치 정밀도]")
        self.results['cam3'] = self.analyze_static_position(
            'cam3', 'cam_3_x(mm)', 'cam_3_y(mm)', 'cam_3_rz(deg)'
        )
        #self._print_static_results('Cam3', self.results['cam3'])
        
        # Cam2 궤적 분석
        print("\n[Cam2 - 궤적 반복 정밀도]")
        self.results['cam2'] = self.analyze_trajectory(sampling_interval_mm)
        #self._print_trajectory_results('Cam2', self.results['cam2'])
    
    def save_results_to_csv(self, output_dir: str = 'outputs'):
        """분석 결과를 CSV 파일로 저장"""
        
        # 1. Summary CSV (전체 통계)
        summary_data = []
        
        # Cam1 데이터
        cam1 = self.results['cam1']
        summary_data.append({
            'Camera': 'Cam1',
            'Type': 'Static',
            'Mean_X(mm)': cam1['mean_x'],
            'Mean_Y(mm)': cam1['mean_y'],
            'Mean_Theta(deg)': cam1['mean_theta'],
            'Sigma_X(mm)': cam1['sigma_x'],
            'Sigma_Y(mm)': cam1['sigma_y'],
            'Sigma_Theta(deg)': cam1['sigma_theta'],
            'Sigma_2D(mm)': cam1['sigma_2d'],
            #'Rp_ISO9283(mm)': cam1['Rp_ISO9283'],
        })
        
        # Cam3 데이터
        cam3 = self.results['cam3']
        summary_data.append({
            'Camera': 'Cam3',
            'Type': 'Static',
            'Mean_X(mm)': cam3['mean_x'],
            'Mean_Y(mm)': cam3['mean_y'],
            'Mean_Theta(deg)': cam3['mean_theta'],
            'Sigma_X(mm)': cam3['sigma_x'],
            'Sigma_Y(mm)': cam3['sigma_y'],
            'Sigma_Theta(deg)': cam3['sigma_theta'],
            'Sigma_2D(mm)': cam3['sigma_2d'],
            #'Rp_ISO9283(mm)': cam3['Rp_ISO9283'],
        })
        
        # Cam2 전체 통계
        cam2 = self.results['cam2']
        summary_data.append({
            'Camera': 'Cam2',
            'Type': 'Trajectory',
            'Mean_X(mm)': np.nan,  # X는 제어 변수
            'Mean_Y(mm)': np.mean(cam2['reference']['y']),
            'Mean_Theta(deg)': np.mean(cam2['reference']['theta']),
            'Sigma_X(mm)': np.nan,  # X는 측정 안함
            'Sigma_Y(mm)': cam2['sigma_y'],
            'Sigma_Theta(deg)': cam2['sigma_theta'],
            'Sigma_2D(mm)': np.nan,  # 궤적은 2D 개념 없음
            #'Rp_ISO9283(mm)': np.nan,
        })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = f'{output_dir}/repeatability_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"\n[Summary CSV 저장] {summary_path}")
        
        # 2. Cam2 Detailed CSV (샘플 포인트별)
        cam2_detailed_data = []
        
        for i, x_pos in enumerate(cam2['target_x']):
            cam2_detailed_data.append({
                'Sample_Index': i,
                'X_Position(mm)': x_pos,
                'Reference_Y(mm)': cam2['reference']['y'][i],
                'Reference_Theta(deg)': cam2['reference']['theta'][i],
                'Sigma_Y(mm)': cam2['sigma_y_at_each_x'][i],
                'Sigma_Theta(deg)': cam2['sigma_theta_at_each_x'][i],
            })
        
        cam2_detailed_df = pd.DataFrame(cam2_detailed_data)
        cam2_detailed_path = f'{output_dir}/cam2_trajectory_detailed.csv'
        cam2_detailed_df.to_csv(cam2_detailed_path, index=False)
        print(f"[Cam2 Detailed CSV 저장] {cam2_detailed_path}")
        
        # 3. Cam1 & Cam3 개별 측정값 (선택사항)
        cam1_measurements = pd.DataFrame({
            'Trial': range(len(self.df)),
            'X(mm)': self.df['cam_1_x(mm)'],
            'Y(mm)': self.df['cam_1_y(mm)'],
            'Theta(deg)': self.df['cam_1_rz(deg)'],
            'Position_Error(mm)': cam1['position_errors'],
            'Theta_Error(deg)': cam1['theta_errors']
        })
        cam1_measurements_path = f'{output_dir}/cam1_measurements.csv'
        cam1_measurements.to_csv(cam1_measurements_path, index=False)
        print(f"[Cam1 Measurements CSV 저장] {cam1_measurements_path}")
        
        cam3_measurements = pd.DataFrame({
            'Trial': range(len(self.df)),
            'X(mm)': self.df['cam_3_x(mm)'],
            'Y(mm)': self.df['cam_3_y(mm)'],
            'Theta(deg)': self.df['cam_3_rz(deg)'],
            'Position_Error(mm)': cam3['position_errors'],
            'Theta_Error(deg)': cam3['theta_errors']
        })
        cam3_measurements_path = f'{output_dir}/cam3_measurements.csv'
        cam3_measurements.to_csv(cam3_measurements_path, index=False)
        print(f"[Cam3 Measurements CSV 저장] {cam3_measurements_path}")
        
        return {
            'summary': summary_path,
            'cam2_detailed': cam2_detailed_path,
            'cam1_measurements': cam1_measurements_path,
            'cam3_measurements': cam3_measurements_path
        }
        
    def _print_static_results(self, cam_name: str, results: Dict):
        """정지 위치 결과 출력"""
        print(f"  평균 위치: ({results['mean_x']:.3f}, {results['mean_y']:.3f}) mm")
        print(f"  평균 각도: {results['mean_theta']:.3f}°")
        print(f"  σ_x: {results['sigma_x']:.4f} mm")
        print(f"  σ_y: {results['sigma_y']:.4f} mm")
        print(f"  σ_θ: {results['sigma_theta']:.4f}°")
        print(f"  σ_2D (position): {results['sigma_2d']:.4f} mm")
        #print(f"  Rp (ISO 9283): {results['Rp_ISO9283']:.4f} mm")
    
    def _print_trajectory_results(self, cam_name: str, results: Dict):
        """궤적 결과 출력"""
        print(f"  X 범위: {results['x_range'][0]:.2f} ~ {results['x_range'][1]:.2f} mm")
        print(f"  샘플링 간격: {results['sampling_interval']:.1f} mm")
        print(f"  샘플 개수: {results['n_samples']}")
        print(f"  σ_y (lateral deviation): {results['sigma_y']:.4f} mm")
        print(f"  σ_θ (angular deviation): {results['sigma_theta']:.4f}°")
        print(f"  최대 Y 표준편차: {np.max(results['sigma_y_at_each_x']):.4f} mm")
        print(f"  최소 Y 표준편차: {np.min(results['sigma_y_at_each_x']):.4f} mm")
    
    def plot_results(self, output_dir: str = 'outputs'):
        """결과 시각화"""
        fig = plt.figure(figsize=(16, 10))
        
        # Cam1 scatter plot
        ax1 = plt.subplot(2, 3, 1)
        cam1_data = self.results['cam1']
        x_data = self.df['cam_1_x(mm)'].values
        y_data = self.df['cam_1_y(mm)'].values
        ax1.scatter(x_data, y_data, alpha=0.6, s=50)
        ax1.plot(cam1_data['mean_x'], cam1_data['mean_y'], 'r*', markersize=15, label='Mean')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_title(f"Cam1 Position\nσ_2D={cam1_data['sigma_2d']:.4f} mm")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Cam3 scatter plot
        ax2 = plt.subplot(2, 3, 2)
        cam3_data = self.results['cam3']
        x_data = self.df['cam_3_x(mm)'].values
        y_data = self.df['cam_3_y(mm)'].values
        ax2.scatter(x_data, y_data, alpha=0.6, s=50)
        ax2.plot(cam3_data['mean_x'], cam3_data['mean_y'], 'r*', markersize=15, label='Mean')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_title(f"Cam3 Position\nσ_2D={cam3_data['sigma_2d']:.4f} mm")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # Cam1 & Cam3 error histogram
        ax3 = plt.subplot(2, 3, 3)
        # Use 'auto' bins to handle cases with limited data range
        cam1_errors = cam1_data['position_errors']
        cam3_errors = cam3_data['position_errors']
        
        # Calculate appropriate bin count based on data
        def get_bins(data, max_bins=20):
            if len(data) < 2:
                return 1
            data_range = np.ptp(data)  # peak-to-peak (max - min)
            if data_range == 0:
                return 1
            return min(max_bins, max(1, len(data) // 5))
        
        cam1_bins = get_bins(cam1_errors)
        cam3_bins = get_bins(cam3_errors)
        
        if len(cam1_errors) > 0:
            ax3.hist(cam1_errors, bins=cam1_bins, alpha=0.5, label='Cam1', edgecolor='black')
        if len(cam3_errors) > 0:
            ax3.hist(cam3_errors, bins=cam3_bins, alpha=0.5, label='Cam3', edgecolor='black')
        ax3.set_xlabel('2D Position Error (mm)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Position Error Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Cam2 trajectories overlay (X-Y plot)
        ax4 = plt.subplot(2, 3, 4)
        cam2_data = self.results['cam2']
        
        # Plot all trial trajectories
        for traj in cam2_data['trajectories']:
            ax4.plot(traj['x'], traj['y'], 'b-', alpha=0.3, linewidth=0.5)
        
        # Plot reference trajectory
        ax4.plot(cam2_data['reference']['x'], cam2_data['reference']['y'], 
                'r-', linewidth=2, label='Reference')
        ax4.set_xlabel('X (mm)')
        ax4.set_ylabel('Y (mm)')
        ax4.set_title(f"Cam2 Trajectories\nσ_y={cam2_data['sigma_y']:.4f} mm")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')
        
        # Y error along X trajectory
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(cam2_data['target_x'], cam2_data['sigma_y_at_each_x'], 'b-', linewidth=2)
        ax5.fill_between(cam2_data['target_x'], 0, cam2_data['sigma_y_at_each_x'], alpha=0.3)
        ax5.set_xlabel('X Position (mm)')
        ax5.set_ylabel('σ_y (mm)')
        ax5.set_title('Cam2 Trajectories\nY Repeatability (Lateral Deviation)')
        ax5.grid(True, alpha=0.3)
        
        # Angular error along X trajectory
        ax6 = plt.subplot(2, 3, 6)
        ax6.plot(cam2_data['target_x'], cam2_data['sigma_theta_at_each_x'], 'g-', linewidth=2)
        ax6.fill_between(cam2_data['target_x'], 0, cam2_data['sigma_theta_at_each_x'], alpha=0.3)
        ax6.set_xlabel('X Position (mm)')
        ax6.set_ylabel('σ_θ (deg)')
        ax6.set_title('Cam2 Trajectories\nRZ Repeatability (Angular Deviation)')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/repeatability_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n[그래프 저장] {output_dir}/repeatability_analysis.png")
        
        return fig


if __name__ == "__main__":
    # 분석 실행
    parser = argparse.ArgumentParser(description="Trajectory Repeatability Analysis")
    parser.add_argument('--csv_path', type=str, default='data/20251118-154122_zoom1_raw_data.csv', help='CSV file path')
    parser.add_argument('--sampling_interval_mm', type=float, default=20.0, help='Sampling interval in mm')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory for CSV files')
    args = parser.parse_args()

    analyzer = TrajectoryRepeatability(args.csv_path)
    analyzer.run_analysis(sampling_interval_mm= args.sampling_interval_mm)
    analyzer.plot_results(output_dir=args.output_dir)
    
    # CSV 결과 저장
    csv_paths = analyzer.save_results_to_csv(args.output_dir)
    
    print("\n" + "=" * 60)
    print("분석 완료!")
    print("=" * 60)