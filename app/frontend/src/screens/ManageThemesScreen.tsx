import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import LinearGradient from 'react-native-linear-gradient';
import Header from '../components/Header';
import NeonButton from '../components/NeonButton';
import { StackNavigationProp } from '@react-navigation/stack';

interface ManageThemesScreenProps {
  navigation: StackNavigationProp<any, any>;
}

const ManageThemesScreen: React.FC<ManageThemesScreenProps> = ({ navigation }) => {
  return (
    <LinearGradient
      colors={["#a4508b", "#5f0a87", "#39ff14"]}
      style={styles.background}
    >
      <Header title="Manage Themes" />
      <View style={styles.container}>
        <Text style={styles.title}>Your Themes</Text>
        {/* Placeholder for theme list */}
        <View style={styles.listPlaceholder}>
          <Text style={styles.placeholderText}>[Theme List Here]</Text>
        </View>
        <NeonButton text="Add Theme" onPress={() => {}} />
        <NeonButton text="Back" onPress={() => navigation.goBack()} />
      </View>
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  background: {
    flex: 1,
  },
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 24,
  },
  title: {
    fontSize: 24,
    color: '#fff',
    fontWeight: 'bold',
    marginBottom: 24,
    letterSpacing: 1,
  },
  listPlaceholder: {
    width: '100%',
    height: 120,
    backgroundColor: '#1a002a',
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 24,
    borderWidth: 2,
    borderColor: '#39ff14',
  },
  placeholderText: {
    color: '#fff',
    fontSize: 16,
    opacity: 0.7,
  },
});

export default ManageThemesScreen; 