import React from 'react';
import { TouchableOpacity, Text, StyleSheet, ViewStyle } from 'react-native';

interface NeonButtonProps {
  text: string;
  onPress: () => void;
  style?: ViewStyle;
  disabled?: boolean;
}

const NeonButton: React.FC<NeonButtonProps> = ({ text, onPress, style, disabled }) => {
  return (
    <TouchableOpacity
      style={[styles.button, style, disabled && styles.disabled]}
      onPress={onPress}
      activeOpacity={0.8}
      accessibilityRole="button"
      accessibilityLabel={text}
      disabled={disabled}
    >
      <Text style={styles.text}>{text}</Text>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    backgroundColor: '#2a003f', // Deep purple
    borderRadius: 24,
    paddingVertical: 14,
    paddingHorizontal: 32,
    alignItems: 'center',
    marginVertical: 10,
    // Neon glow effect
    shadowColor: '#ff00cc', // Neon pink
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.7,
    shadowRadius: 12,
    elevation: 10,
    borderWidth: 2,
    borderColor: '#39ff14', // Neon green border
  },
  text: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
    letterSpacing: 1,
    textShadowColor: '#39ff14',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 6,
  },
  disabled: {
    opacity: 0.5,
  },
});

export default NeonButton; 