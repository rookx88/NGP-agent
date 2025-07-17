import React, { useContext } from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import LatestTrendingScreen from '../screens/LatestTrendingScreen';
import PeepsOfInterestScreen from '../screens/PeepsOfInterestScreen';
import YourWorldScreen from '../screens/YourWorldScreen';
import { AuthContext } from '../context/AuthContext';
// import Icon from 'react-native-vector-icons/MaterialCommunityIcons'; // Example icon usage

const Tab = createBottomTabNavigator();

const MainTab: React.FC = () => {
  const { userName } = useContext(AuthContext);
  return (
    <Tab.Navigator
      initialRouteName="LatestTrending"
      screenOptions={{
        headerShown: false,
        // tabBarIcon: ({ color, size }) => (<Icon name="home" color={color} size={size} />),
        tabBarStyle: { backgroundColor: '#2a003f' }, // Example dark background
        tabBarActiveTintColor: '#39ff14', // Neon green
        tabBarInactiveTintColor: '#fff',
      }}
    >
      <Tab.Screen name="LatestTrending" component={LatestTrendingScreen} options={{ title: 'Trending' }} />
      <Tab.Screen name="PeepsOfInterest" component={PeepsOfInterestScreen} options={{ title: 'Peeps' }} />
      <Tab.Screen name="YourWorld" component={YourWorldScreen} options={{ title: userName ? userName : 'Your World' }} />
    </Tab.Navigator>
  );
};

export default MainTab; 