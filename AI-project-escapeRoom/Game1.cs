using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using System.Linq;
using System;
using System.Threading.Tasks;
using System.Threading;
using System.Collections.Generic;

namespace AI_project_escapeRoom
{
    public class Game1 : Game
    {
        private GraphicsDeviceManager _graphics;
        private SpriteBatch _spriteBatch;

        // Training components
        private GameEnvironment gameEnvironment;
        private PPO ppo;
        private Task trainingTask;
        private CancellationTokenSource cancellationSource;
        private bool isTraining = true;
        private readonly object gameLock = new object();

        // Game Objects
        public Player player;
        public Box box;
        public Wall ground;
        public Wall platform;
        public Wall button;
        public Wall wall;
        public Wall door;
        public Wall cieling;
        public bool IsPressed = false;
        public bool IsOpen = false;
        public Wall[] walls;

        // Environment Settings
        public float groundLevel;
        public float widthLevel;

        // Camera Settings
        public Vector2 cameraPosition;
        public int ScreenWidth = 1280;
        public int ScreenHeight = 720;

        // Helper variables for new functions
        public Vector2 lastPlayerPosition;
        private int idleSteps = 0;
        private HashSet<Point> visitedAreas = new HashSet<Point>();
        public bool previousBoxState = false;


        public Game1()
        {
            _graphics = new GraphicsDeviceManager(this);
            Content.RootDirectory = "Content";
            IsMouseVisible = true;

            _graphics.PreferredBackBufferWidth = 1280;
            _graphics.PreferredBackBufferHeight = 720;
            _graphics.ApplyChanges();

            cameraPosition = Vector2.Zero;
        }

        protected override void Initialize()
        {
            base.Initialize();

            groundLevel = _graphics.PreferredBackBufferHeight - 50;
            widthLevel = _graphics.PreferredBackBufferWidth - 50;

            InitializePlayer();
            InitializeEnvironment();

            // Initialize training components
            gameEnvironment = new GameEnvironment(this);
            ppo = new PPO();

            if (isTraining)
            {
                StartTraining();
            }
        }

        // New Function: IsIdle
        public bool IsIdle()
        {
            if (player.Position == lastPlayerPosition)
            {
                idleSteps++;
            }
            else
            {
                idleSteps = 0; // Reset idle steps if the player moves
                lastPlayerPosition = player.Position;
            }

            // Consider idle if no movement for more than 30 steps
            return idleSteps > 30;
        }

        // New Function: IsExploringNewArea
        public bool IsExploringNewArea()
        {
            // Define a grid-like area system for simplicity
            Point currentArea = new Point(
                (int)(player.Position.X / 50), // Divide by tile or grid size
                (int)(player.Position.Y / 50)
            );

            // Check if the area has been visited
            if (!visitedAreas.Contains(currentArea))
            {
                visitedAreas.Add(currentArea); // Mark the area as visited
                return true; // Player is exploring a new area
            }

            return false; // Area already visited
        }

        // New Function: IsMovingToward
        public bool IsMovingToward(GameObject target, Vector2 previousPlayerPosition)
        {
            // Get the direction vectors
            Vector2 directionToTarget = target.Position - previousPlayerPosition;
            Vector2 movementDirection = player.Position - previousPlayerPosition;

            // Early exit if no movement
            if (movementDirection.LengthSquared() < 0.0001f) return false;

            // Avoid normalization when not needed
            float dotProduct = Vector2.Dot(directionToTarget, movementDirection) /
                            (directionToTarget.Length() * movementDirection.Length());

            // Ensure the dot product is meaningful
            return dotProduct > 0.95f; // Slightly more strict
        }


        private void StartTraining()
        {
            cancellationSource = new CancellationTokenSource();
            trainingTask = Task.Run(() => TrainingLoop(cancellationSource.Token));
        }

        private async Task TrainingLoop(CancellationToken token)
        {
            try
            {
                await Task.Run(() =>
                {
                    ppo.Train(gameEnvironment, 1000000, "ppo_prog", "ppo_model.meta");
                }, token);

                // Switch to manual mode after training
                Console.WriteLine("Training completed.");
                isTraining = false;
            }
            catch (OperationCanceledException)
            {
                // Training was cancelled
                isTraining = false;
            }
        }

        // Your existing initialization methods remain the same
        private void InitializePlayer()
        {
            Vector2 playerSize = new Vector2(50, 50);
            Vector2 playerStartPosition = new Vector2(100, groundLevel - playerSize.Y);
            player = new Player(playerStartPosition, playerSize, "PLAYER");
            player.LoadTexture(GraphicsDevice, Color.Red);
        }
        private void InitializeEnvironment()
        {
            // Initialize box
            Vector2 boxSize = new Vector2(50, 50);
            Vector2 boxStartPosition = new Vector2(200, groundLevel - boxSize.Y);
            box = new Box(boxStartPosition, boxSize);
            box.LoadTexture(GraphicsDevice, Color.Blue);

            // Initialize ground
            Vector2 groundSize = new Vector2(10000, 50);
            Vector2 groundPosition = new Vector2(0, groundLevel);
            ground = new Wall(groundPosition, groundSize);
            ground.LoadTexture(GraphicsDevice, Color.Black);

            // Initialize platform
            Vector2 platformSize = new Vector2(500, 25);
            Vector2 platformPosition = new Vector2(500, groundLevel - 100);
            platform = new Wall(platformPosition, platformSize);
            platform.LoadTexture(GraphicsDevice, Color.White);

            // Initialize button
            Vector2 buttonSize = new Vector2(100, 10);
            Vector2 buttonPosition = new Vector2(500, groundLevel - 110);
            button = new Wall(buttonPosition, buttonSize) { ROLL = "BUTTON" };
            button.LoadTexture(GraphicsDevice, Color.Red);

            // Initialize wall
            Vector2 wallSize = new Vector2(50, 1000);
            Vector2 wallPosition = new Vector2(0, 0);
            wall = new Wall(wallPosition, wallSize);
            wall.LoadTexture(GraphicsDevice, Color.Black);

            // Initialize door
            Vector2 doorSize = new Vector2(50, 1000);
            Vector2 doorPosition = new Vector2(widthLevel, 0);
            door = new Wall(doorPosition, doorSize);
            door.LoadTexture(GraphicsDevice, Color.Pink);

            // Initialize cieling
            Vector2 cielingSize = new Vector2(10000, 50);
            Vector2 cielingPosition = new Vector2(0, 0);
            cieling = new Wall(cielingPosition, cielingSize);
            cieling.LoadTexture(GraphicsDevice, Color.Black);

            // Group walls
            walls = new Wall[] { ground, platform, wall, door, cieling };
        }

        protected override void LoadContent()
        {
            _spriteBatch = new SpriteBatch(GraphicsDevice);
        }

        protected override void Update(GameTime gameTime)
        {
            if (GamePad.GetState(PlayerIndex.One).Buttons.Back == ButtonState.Pressed ||
                Keyboard.GetState().IsKeyDown(Keys.Escape))
            {
                Exit();
            }

            lock (gameLock)
            {
                if (!isTraining)
                {
                    // Only handle manual input when not training
                    HandleManualInput();
                }

                HandleCollisions();
                HandleCameraMovement();

                // Update game objects
                player.Update(gameTime);
                box.Update(gameTime);
                ground.Update(gameTime);
                platform.Update(gameTime);
                button.Update(gameTime);
                wall.Update(gameTime);
                door.Update(gameTime);
                cieling.Update(gameTime);

                // Button interaction logic
                UpdateButtonState();

            }

            base.Update(gameTime);
        }

        private void HandleManualInput()
        {
            KeyboardState state = Keyboard.GetState();

            if (state.IsKeyDown(Keys.A)) player.Move(new Vector2(-10, 0));
            if (state.IsKeyDown(Keys.D)) player.Move(new Vector2(10, 0));
            if (state.IsKeyDown(Keys.Space) && player.IsGrounded)
            {
                player.ApplyForce(new Vector2(0, -250));
                player.IsGrounded = false;
            }
            if (state.IsKeyDown(Keys.E)) player.Grab(box);
            if (state.IsKeyDown(Keys.Q)) player.DropHeldBox();
        }

        private void UpdateButtonState()
        {
            if (player.Intersects(button) || box.Intersects(button))
            {
                button.LoadTexture(GraphicsDevice, Color.Green);
                IsPressed = true;
                IsOpen = true;
                door.Position = new Vector2(99999, 999999);
            }
            else
            {
                button.LoadTexture(GraphicsDevice, Color.Red);
                IsPressed = false;
                IsOpen = false;
                door.Position = new Vector2(widthLevel, 0);
            }
        }

        // Your existing methods remain the same
        private void HandleCameraMovement()
        {
            // Check if the player is out of the screen bounds
            if (player.Position.X < cameraPosition.X) // Move camera left
            {
                cameraPosition.X -= ScreenWidth;
            }
            else if (player.Position.X > cameraPosition.X + ScreenWidth) // Move camera right
            {
                cameraPosition.X += ScreenWidth;
            }

            if (player.Position.Y < cameraPosition.Y) // Move camera up
            {
                cameraPosition.Y -= ScreenHeight;
            }
            else if (player.Position.Y > cameraPosition.Y + ScreenHeight) // Move camera down
            {
                cameraPosition.Y += ScreenHeight;
            }
        }
        private void HandleCollisions()
        {
            player.StopByWalls(walls);
            box.StopByWalls(walls);
            button.StopByWalls(walls);
        }

        protected override void Draw(GameTime gameTime)
        {
            GraphicsDevice.Clear(Color.DarkBlue);

            _spriteBatch.Begin(transformMatrix: Matrix.CreateTranslation(-cameraPosition.X, -cameraPosition.Y, 0));

            // Draw game objects
            player.Draw(_spriteBatch);
            box.Draw(_spriteBatch);
            ground.Draw(_spriteBatch);
            platform.Draw(_spriteBatch);
            button.Draw(_spriteBatch);
            wall.Draw(_spriteBatch);
            door.Draw(_spriteBatch);
            cieling.Draw(_spriteBatch);


            // Draw text at the upper corner of the screen
            SpriteFont font = Content.Load<SpriteFont>("File");
            string title = "Escape Room AI";
            Vector2 textPosition = new Vector2(10, 10);
            _spriteBatch.DrawString(font, title, textPosition, Color.Red);
            _spriteBatch.DrawString(font, "Epesode:  " + ppo.curentEpeisode, new Vector2(10, 30), Color.Red);
            _spriteBatch.DrawString(font, "Rewards:  " + ppo.episodeRewards.Sum().ToString(), new Vector2(10, 50), Color.Red);
            _spriteBatch.DrawString(font, "Policy loss:  " + ppo.policyLossesfordispaly.ToString("F3"), new Vector2(10, 70), Color.Red);
            _spriteBatch.DrawString(font, "value loss:  " + ppo.Value_Loss.ToString("F3"), new Vector2(10, 90), Color.Red);
            _spriteBatch.DrawString(font, "Entropy:  " + ppo.Entropy.ToString("F3"), new Vector2(10, 110), Color.Red);
            base.Draw(gameTime);
            _spriteBatch.End();
        }

        protected override void UnloadContent()
        {
            cancellationSource?.Cancel();
            trainingTask?.Wait();
            base.UnloadContent();
        }
    }
}