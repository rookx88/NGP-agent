import React from 'react';
import { View, TextInput, StyleSheet, TextInputProps, ViewStyle } from 'react-native';
// import Icon from 'react-native-vector-icons/MaterialCommunityIcons'; // For input icons

interface NeonInputProps extends TextInputProps {
  value: string;
  onChangeText: (text: string) => void;
  placeholder?: string;
  secureTextEntry?: boolean;
  icon?: string;
  keyboardType?: TextInputProps['keyboardType'];
  style?: ViewStyle;
}

const NeonInput: React.FC<NeonInputProps> = ({
  value,
  onChangeText,
  placeholder,
  secureTextEntry,
  icon, // Pass icon name as string
  keyboardType = 'default',
  style,
  ...props
}) => {
  return (
    <View style={styles.container}>
      {/* Uncomment and use Icon if available */}
      {/* {icon && <Icon name={icon} size={24} color="#fff" style={styles.icon} />} */}
      <TextInput
        style={[styles.input, style]}
        value={value}
        onChangeText={onChangeText}
        placeholder={placeholder}
        placeholderTextColor="#ccc"
        secureTextEntry={secureTextEntry}
        keyboardType={keyboardType}
        accessibilityLabel={placeholder}
        {...props}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1a002a',
    borderRadius: 18,
    borderWidth: 2,
    borderColor: '#ff00cc', // Neon pink border
    marginVertical: 8,
    paddingHorizontal: 12,
    // Neon glow effect
    shadowColor: '#39ff14',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.5,
    shadowRadius: 8,
    elevation: 6,
  },
  icon: {
    marginRight: 8,
  },
  input: {
    flex: 1,
    color: '#fff',
    fontSize: 16,
    paddingVertical: 12,
    fontFamily: 'sans-serif',
  },
});

export default NeonInput; 