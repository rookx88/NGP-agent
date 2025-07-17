import React, { createContext, useState, useEffect, ReactNode } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as api from '../api/api'; // Placeholder API functions

interface AuthContextType {
  isAuthenticated: boolean;
  userToken: string | null;
  userName: string;
  userEmail: string;
  loading: boolean;
  error: string | null;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

export const AuthContext = createContext<AuthContextType>({} as AuthContextType);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userToken, setUserToken] = useState<string | null>(null);
  const [userName, setUserName] = useState('');
  const [userEmail, setUserEmail] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // On mount, check for stored token
    const loadToken = async () => {
      try {
        const token = await AsyncStorage.getItem('userToken');
        const name = await AsyncStorage.getItem('userName');
        const email = await AsyncStorage.getItem('userEmail');
        if (token) {
          setUserToken(token);
          setUserName(name || '');
          setUserEmail(email || '');
          setIsAuthenticated(true);
        }
      } catch (e) {
        setError('Failed to load user data');
      } finally {
        setLoading(false);
      }
    };
    loadToken();
  }, []);

  const login = async (username: string, password: string) => {
    setLoading(true);
    setError(null);
    try {
      // Placeholder API call
      const { token, name, email } = await api.login(username, password);
      await AsyncStorage.setItem('userToken', token);
      await AsyncStorage.setItem('userName', name);
      await AsyncStorage.setItem('userEmail', email);
      setUserToken(token);
      setUserName(name);
      setUserEmail(email);
      setIsAuthenticated(true);
    } catch (e: any) {
      setError(e.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  const register = async (username: string, email: string, password: string) => {
    setLoading(true);
    setError(null);
    try {
      // Placeholder API call
      const { token, name } = await api.register(username, email, password);
      await AsyncStorage.setItem('userToken', token);
      await AsyncStorage.setItem('userName', name);
      await AsyncStorage.setItem('userEmail', email);
      setUserToken(token);
      setUserName(name);
      setUserEmail(email);
      setIsAuthenticated(true);
    } catch (e: any) {
      setError(e.message || 'Registration failed');
    } finally {
      setLoading(false);
    }
  };

  const logout = async () => {
    setLoading(true);
    try {
      await AsyncStorage.removeItem('userToken');
      await AsyncStorage.removeItem('userName');
      await AsyncStorage.removeItem('userEmail');
      setUserToken(null);
      setUserName('');
      setUserEmail('');
      setIsAuthenticated(false);
    } catch (e) {
      setError('Logout failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthContext.Provider
      value={{
        isAuthenticated,
        userToken,
        userName,
        userEmail,
        loading,
        error,
        login,
        register,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}; 