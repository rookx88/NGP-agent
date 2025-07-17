// Placeholder API functions for Synch app
// Replace with real backend integration

export interface LoginResponse {
  token: string;
  name: string;
  email: string;
}

export interface RegisterResponse {
  token: string;
  name: string;
}

export const login = async (username: string, password: string): Promise<LoginResponse> => {
  // Simulate network delay
  await new Promise((resolve) => setTimeout(resolve, 1000));
  if (username === 'test' && password === 'password') {
    return {
      token: 'mock-token-123',
      name: 'Test User',
      email: 'test@example.com',
    };
  } else {
    throw new Error('Invalid credentials');
  }
};

export const register = async (username: string, email: string, password: string): Promise<RegisterResponse> => {
  // Simulate network delay
  await new Promise((resolve) => setTimeout(resolve, 1000));
  if (username && email && password) {
    return {
      token: 'mock-token-456',
      name: username,
    };
  } else {
    throw new Error('Registration failed');
  }
};

// Add more placeholder functions for trend analysis and influencer engagement as needed 