interface TabBarProps {
  tabs: string[];
  activeTab: string;
  onTabChange: (tab: string) => void;
}

const TabBar = ({ tabs, activeTab, onTabChange }: TabBarProps) => {
  return (
    <div className="flex border-b border-gray-200 bg-white px-6">
      {tabs.map((tab) => (
        <button
          key={tab}
          onClick={() => onTabChange(tab)}
          className={`px-5 py-3 text-sm font-medium transition-colors relative ${
            activeTab === tab
              ? 'text-intel-dark after:absolute after:bottom-0 after:left-0 after:right-0 after:h-[2px] after:bg-intel-blue'
              : 'text-intel-gray hover:text-intel-dark'
          }`}
        >
          {tab}
        </button>
      ))}
    </div>
  );
};

export default TabBar;
