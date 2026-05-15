import BrandSlot from '../../assets/BrandSlot.svg';
import { constants } from '../../constants';

const Header = () => {
  return (
    <header className="sticky top-0 left-0 right-0 z-50 bg-intel-blue w-full h-[72px] flex items-center px-8 border-b border-intel-blue-dark">
      <div className="flex items-center gap-4">
        <img src={BrandSlot} alt="Intel" className="w-[89px] h-[72px] object-contain" />
        <span className="text-lg font-medium text-white font-display">{constants.TITLE}</span>
      </div>
    </header>
  );
};

export default Header;
