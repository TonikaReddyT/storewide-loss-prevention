import type { TrackingUpdate } from '../../types';

interface Props {
  tracking: TrackingUpdate;
  onClose: () => void;
}

const fmt = (ts: string | null) =>
  ts ? new Date(ts).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : 'now';

const fmtDuration = (sec: number) => {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
};

const TrackingPanel = ({ tracking, onClose }: Props) => {
  const traj = tracking.trajectory;

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={onClose}>
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-lg mx-4 max-h-[85vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-display font-medium text-intel-dark">POI Tracking — {tracking.poi_id}</h2>
            <p className="text-xs text-intel-gray">Alert: {tracking.alert_id}</p>
          </div>
          <button onClick={onClose} className="w-8 h-8 rounded-full hover:bg-gray-100 flex items-center justify-center text-intel-gray">✕</button>
        </div>

        <div className="p-6 space-y-5">
          {/* Summary */}
          <div className="flex gap-4">
            <div className="flex-1 bg-blue-50 rounded-lg p-3 text-center">
              <p className="text-2xl font-semibold text-intel-blue">{fmtDuration(tracking.time_in_store_sec)}</p>
              <p className="text-[11px] text-intel-gray mt-0.5">Time in Store</p>
            </div>
            <div className="flex-1 bg-blue-50 rounded-lg p-3 text-center">
              <p className="text-2xl font-semibold text-intel-blue">{traj.length}</p>
              <p className="text-[11px] text-intel-gray mt-0.5">Cameras Visited</p>
            </div>
            <div className="flex-1 bg-blue-50 rounded-lg p-3 text-center">
              <p className="text-sm font-semibold text-intel-blue">{tracking.current_location.camera_id}</p>
              <p className="text-[11px] text-intel-gray mt-0.5">Current Location</p>
              <p className="text-[10px] text-intel-gray">{(tracking.current_location.confidence * 100).toFixed(0)}% confidence</p>
            </div>
          </div>

          {/* Trajectory timeline */}
          <div>
            <p className="text-xs font-medium text-intel-gray mb-3">Trajectory</p>
            <div className="relative pl-6 space-y-4">
              {/* Vertical line */}
              <div className="absolute left-[9px] top-1 bottom-1 w-px bg-intel-blue/30" />

              {traj.map((step, i) => {
                const isCurrent = step.last_seen === null;
                return (
                  <div key={i} className="relative flex items-start gap-3">
                    {/* Dot */}
                    <div className={`absolute -left-6 top-0.5 w-[18px] h-[18px] rounded-full border-2 flex items-center justify-center ${
                      isCurrent ? 'border-intel-blue bg-intel-blue' : 'border-intel-blue/60 bg-white'
                    }`}>
                      {isCurrent && <div className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />}
                    </div>

                    <div className="flex-1 bg-gray-50 rounded-lg p-3">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-intel-dark">{step.camera_id}</span>
                        {isCurrent && <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-green-100 text-green-700 animate-pulse">LIVE</span>}
                      </div>
                      <p className="text-xs text-intel-gray mt-1">
                        {fmt(step.first_seen)} → {fmt(step.last_seen)}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrackingPanel;
