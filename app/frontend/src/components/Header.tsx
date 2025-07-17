import React from 'react';
import { View, Text, Image, StyleSheet, TouchableOpacity, GestureResponderEvent } from 'react-native';
// import Icon from 'react-native-vector-icons/MaterialCommunityIcons'; // For avatar icon

interface HeaderProps {
  title?: string;
  userName?: string;
  onAvatarPress?: (event: GestureResponderEvent) => void;
}

const Header: React.FC<HeaderProps> = ({ title, userName, onAvatarPress }) => {
  return (
    <View style={styles.container}>
      {/* App Logo (left) */}
      <Image
        source={require('../assets/logo.png')} // PLACEHOLDER: Replace with actual logo asset
        style={styles.logo}
        accessibilityLabel="Synch Logo"
      />
      {/* Title or Username (center) */}
      <Text style={styles.title}>{title || userName || 'Synch'}</Text>
      {/* User Avatar (right) */}
      <TouchableOpacity onPress={onAvatarPress} style={styles.avatarContainer}>
        {/* Replace with user profile image if available */}
        {/* <Icon name="account-circle" size={36} color="#fff" /> */}
        <Image
          source={require('../assets/user_silhouette.png')} // PLACEHOLDER: Replace with actual user avatar asset
          style={styles.avatar}
          accessibilityLabel="User Avatar"
        />
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: 'transparent',
    // Neon glow effect (simulate with shadow)
    shadowColor: '#39ff14',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.5,
    shadowRadius: 8,
    elevation: 8,
  },
  logo: {
    width: 48,
    height: 48,
    resizeMode: 'contain',
    borderRadius: 24,
    backgroundColor: '#fff1', // Subtle background for contrast
  },
  title: {
    flex: 1,
    textAlign: 'center',
    fontSize: 22,
    fontWeight: 'bold',
    color: '#fff',
    fontFamily: 'sans-serif', // Modern font
    letterSpacing: 1,
    textShadowColor: '#39ff14',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 6,
  },
  avatarContainer: {
    borderRadius: 20,
    overflow: 'hidden',
    borderWidth: 2,
    borderColor: '#39ff14', // Neon green border
    shadowColor: '#39ff14',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.7,
    shadowRadius: 8,
    elevation: 8,
  },
  avatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#fff1',
  },
});

export default Header; 