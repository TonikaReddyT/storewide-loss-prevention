interface ImagePreviewModalProps {
  imageUrl: string;
  alt?: string;
  onClose: () => void;
}

const ImagePreviewModal = ({ imageUrl, alt = 'Preview', onClose }: ImagePreviewModalProps) => {
  return (
    <div
      className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="relative bg-white rounded-xl shadow-2xl p-2 max-w-[90vw] max-h-[90vh]"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          className="absolute -top-3 -right-3 w-8 h-8 bg-intel-dark text-white rounded-full flex items-center justify-center text-lg hover:bg-red-600 transition-colors"
        >
          ×
        </button>
        <img
          src={imageUrl}
          alt={alt}
          className="max-w-[85vw] max-h-[85vh] rounded-lg object-contain"
        />
      </div>
    </div>
  );
};

export default ImagePreviewModal;
