import type { POI } from '../../types';

interface Props {
  poi: POI;
  onClose: () => void;
  onImageClick: (url: string) => void;
}

const severityColor = (s: POI['severity']) =>
  s === 'high' ? 'bg-red-100 text-red-700' : s === 'medium' ? 'bg-yellow-100 text-yellow-700' : 'bg-gray-100 text-gray-600';

const POIDetailModal = ({ poi, onClose, onImageClick }: Props) => {
  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={onClose}>
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-2xl mx-4 max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-display font-medium text-intel-dark">{poi.poi_id}</h2>
            <p className="text-xs text-intel-gray mt-0.5">Enrolled {new Date(poi.timestamp).toLocaleString()}</p>
          </div>
          <button onClick={onClose} className="w-8 h-8 rounded-full hover:bg-gray-100 flex items-center justify-center text-intel-gray transition-colors">✕</button>
        </div>

        <div className="p-6 space-y-5">
          {/* Metadata row */}
          <div className="flex flex-wrap gap-3 text-xs">
            <span className={`px-2.5 py-1 rounded-full font-semibold capitalize ${severityColor(poi.severity)}`}>
              Severity: {poi.severity}
            </span>
            <span className={`px-2.5 py-1 rounded-full font-medium capitalize ${poi.status === 'active' ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'}`}>
              {poi.status}
            </span>
            <span className="px-2.5 py-1 rounded-full bg-blue-50 text-intel-blue font-medium">
              Enrolled by: {poi.enrolled_by}
            </span>
          </div>

          {/* Notes */}
          {poi.notes && (
            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs font-medium text-intel-gray mb-1">Notes</p>
              <p className="text-sm text-intel-dark">{poi.notes}</p>
            </div>
          )}

          {/* Reference Images */}
          <div>
            <p className="text-xs font-medium text-intel-gray mb-2">Reference Images ({poi.reference_images.length})</p>
            <div className="grid grid-cols-3 sm:grid-cols-4 gap-3">
              {poi.reference_images.map((ri, i) => (
                <div key={ri.embedding_id} className="space-y-1">
                  <img
                    src={ri.image_path}
                    alt={`Reference ${i + 1}`}
                    onClick={() => onImageClick(ri.image_path)}
                    className="w-full aspect-[3/4] object-cover rounded-lg border border-gray-200 cursor-pointer hover:ring-2 hover:ring-intel-blue transition-all"
                  />
                  <div className="text-[10px] text-intel-gray text-center space-y-0.5">
                    <p className="truncate" title={ri.embedding_id}>{ri.embedding_id}</p>
                    <p>{ri.vector_dim}-dim · {ri.source}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default POIDetailModal;
