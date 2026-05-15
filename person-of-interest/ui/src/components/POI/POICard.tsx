import type { POI } from '../../types';

interface POICardProps {
  poi: POI;
  onDelete: (id: string) => void;
  onViewDetails: (poi: POI) => void;
  onImageClick: (url: string) => void;
}

const severityBadge = (s: POI['severity']) => {
  const c = s === 'high' ? 'bg-red-100 text-red-700' : s === 'medium' ? 'bg-yellow-100 text-yellow-700' : 'bg-gray-100 text-gray-600';
  return <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold capitalize ${c}`}>{s}</span>;
};

const POICard = ({ poi, onDelete, onViewDetails, onImageClick }: POICardProps) => {
  const primary = poi.reference_images[0]?.image_path ?? '';
  const imgCount = poi.reference_images.length;

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden hover:shadow-md transition-shadow">
      {/* Primary image with count badge */}
      <div className="relative cursor-pointer" onClick={() => primary && onImageClick(primary)}>
        <img src={primary} alt={poi.poi_id} className="w-full h-48 object-cover bg-gray-100" />
        {imgCount > 1 && (
          <span className="absolute top-2 right-2 px-2 py-0.5 rounded-full bg-black/60 text-white text-[10px] font-semibold">
            {imgCount} images
          </span>
        )}
        {/* Thumbnail strip */}
        {imgCount > 1 && (
          <div className="absolute bottom-0 left-0 right-0 flex gap-1 p-1.5 bg-gradient-to-t from-black/60 to-transparent">
            {poi.reference_images.slice(0, 4).map((ri, i) => (
              <img
                key={ri.embedding_id}
                src={ri.image_path}
                alt={`ref-${i}`}
                className="w-8 h-8 rounded object-cover border border-white/60"
                onClick={(e) => { e.stopPropagation(); onImageClick(ri.image_path); }}
              />
            ))}
            {imgCount > 4 && <span className="w-8 h-8 rounded bg-black/50 text-white text-[10px] flex items-center justify-center">+{imgCount - 4}</span>}
          </div>
        )}
      </div>

      <div className="p-4 space-y-2">
        <div className="flex items-center justify-between gap-1">
          <h3 className="font-medium text-intel-dark text-sm truncate">{poi.poi_id}</h3>
          {severityBadge(poi.severity)}
        </div>
        <p className="text-xs text-intel-gray truncate" title={poi.notes}>{poi.notes}</p>
        <div className="flex items-center gap-2 text-[11px] text-intel-gray">
          <span className={`w-1.5 h-1.5 rounded-full ${poi.status === 'active' ? 'bg-green-500' : 'bg-gray-400'}`} />
          <span className="capitalize">{poi.status}</span>
          <span className="mx-1">·</span>
          <span>{new Date(poi.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</span>
        </div>
        <div className="flex gap-2 pt-2">
          <button
            onClick={() => onDelete(poi.poi_id)}
            className="flex-1 px-3 py-1.5 text-xs font-medium text-red-600 bg-red-50 rounded-lg hover:bg-red-100 transition-colors"
          >
            Delete
          </button>
          <button
            onClick={() => onViewDetails(poi)}
            className="flex-1 px-3 py-1.5 text-xs font-medium text-intel-blue bg-blue-50 rounded-lg hover:bg-blue-100 transition-colors"
          >
            View Details
          </button>
        </div>
      </div>
    </div>
  );
};

export default POICard;
