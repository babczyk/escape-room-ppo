using System.Runtime.Serialization.Formatters;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using System.Linq;

namespace AI_project_escapeRoom;

public class Game1 : Game
{
    private GraphicsDeviceManager _graphics;
    private SpriteBatch _spriteBatch;

    // Game Objects
    private Player player;
    private Box box;
    private Wall ground;
    private Wall platform;
    private Wall button;
    private Wall wall;
    private Wall door;
    private Wall cieling;
    private Wall[] walls;

    // Environment Settings
    private float groundLevel;
    private float widthLevel;

    // Camera Settings
    private Vector2 cameraPosition;
    private const int ScreenWidth = 1280;
    private const int ScreenHeight = 720;

    public Game1()
    {
        _graphics = new GraphicsDeviceManager(this);
        Content.RootDirectory = "Content";
        IsMouseVisible = true;

        // Set screen resolution
        _graphics.PreferredBackBufferWidth = 1280;
        _graphics.PreferredBackBufferHeight = 720;
        _graphics.ApplyChanges();

        cameraPosition = Vector2.Zero;

    }

    protected override void Initialize()
    {
        base.Initialize();

        // Set up levels
        groundLevel = _graphics.PreferredBackBufferHeight - 50;
        widthLevel = _graphics.PreferredBackBufferWidth - 50;

        // Initialize game objects
        InitializePlayer();
        InitializeEnvironment();
    }

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
        // Exit game
        if (GamePad.GetState(PlayerIndex.One).Buttons.Back == ButtonState.Pressed ||
            Keyboard.GetState().IsKeyDown(Keys.Escape))
        {
            Exit();
        }

        HandleInput();
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

        base.Update(gameTime);
    }

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

    private void HandleInput()
    {
        KeyboardState state = Keyboard.GetState();

        // Player movement
        if (state.IsKeyDown(Keys.A)) player.Move(new Vector2(-10, 0)); // Move left
        if (state.IsKeyDown(Keys.D)) player.Move(new Vector2(10, 0));  // Move right
        if (state.IsKeyDown(Keys.Space) && player.IsGrounded)
        {
            player.ApplyForce(new Vector2(0, -250)); // Jump
            player.IsGrounded = false;
        }

        // Box interaction
        if (state.IsKeyDown(Keys.E)) player.Grab(box);
        if (state.IsKeyDown(Keys.Q)) player.DropHeldBox();

        // Button interaction
        if (player.Intersects(button) || box.Intersects(button))
        {
            button.LoadTexture(GraphicsDevice, Color.Green);

            door.Position = new Vector2(99999, 999999);
        }
        else
        {
            button.LoadTexture(GraphicsDevice, Color.Red);
            door.Position = new Vector2(widthLevel, 0);
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
        GraphicsDevice.Clear(Color.CornflowerBlue);

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

        _spriteBatch.End();

        base.Draw(gameTime);
    }
}
