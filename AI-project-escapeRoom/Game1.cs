using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;


namespace AI_project_escapeRoom;

public class Game1 : Game
{
    private GraphicsDeviceManager _graphics;
    private SpriteBatch _spriteBatch;

    private Player player; //game object

    //room environment//
    private Box box; //game object
    private Wall ground; //game object
    private Wall platform; //game object
    private Wall[] walls; //game objects
    private float groundLevel;

    public Game1()
    {
        _graphics = new GraphicsDeviceManager(this);
        Content.RootDirectory = "Content";
        IsMouseVisible = true;
    }

    protected override void Initialize()
    {
        base.Initialize();

        // Set up ground level (screen height - some margin)
        groundLevel = _graphics.PreferredBackBufferHeight - 50;

        // Initialize the player GameObject
        Vector2 playerSize = new Vector2(50, 50); // Width and height
        Vector2 playerStartPosition = new Vector2(100, groundLevel - playerSize.Y); // Start position

        player = new Player(playerStartPosition, playerSize, "PLAYER");
        player.LoadTexture(GraphicsDevice, Color.Red); // Give it a red color


        Vector2 boxSize = new Vector2(50, 50); // Width and height
        Vector2 boxStartPosition = new Vector2(200, groundLevel - boxSize.Y); // Start position

        box = new Box(boxStartPosition, boxSize);
        box.LoadTexture(GraphicsDevice, Color.Blue); // Give it a blue color

        Vector2 groundsize = new Vector2(10000, 50); // Width and height
        Vector2 groundposition = new Vector2(0, groundLevel); // Start position

        ground = new Wall(groundposition, groundsize);
        ground.LoadTexture(GraphicsDevice, Color.White); // Give it a white color

        Vector2 platformsize = new Vector2(500, 25); // Width and height
        Vector2 platformposition = new Vector2(500, groundLevel - 100); // Start position

        platform = new Wall(platformposition, platformsize);
        platform.LoadTexture(GraphicsDevice, Color.White); // Give it a white color

        walls = new Wall[] { ground, platform };

    }

    protected override void LoadContent()
    {
        _spriteBatch = new SpriteBatch(GraphicsDevice);
    }

    protected override void Update(GameTime gameTime)
    {

        if (GamePad.GetState(PlayerIndex.One).Buttons.Back == ButtonState.Pressed || Keyboard.GetState().IsKeyDown(Keys.Escape))
            Exit();
        ///
        /// Moovement of player later need to be implimanted through AI (array of moves)
        ///
        // Handle input for moving the player
        KeyboardState state = Keyboard.GetState();
        if (state.IsKeyDown(Keys.A))
        {
            player.Move(new Vector2(-10, 0)); // Move left
        }
        if (state.IsKeyDown(Keys.D))
        {
            player.Move(new Vector2(10, 0)); // Move right
        }
        if (state.IsKeyDown(Keys.Space) && player.IsGrounded)
        {
            player.ApplyForce(new Vector2(0, -250)); // Jump
            player.IsGrounded = false;
        }
        if (state.IsKeyDown(Keys.E))
        {
            player.Grab(box);
        }
        if (state.IsKeyDown(Keys.Q))
        {
            player.DropHeldBox();
        }

        // Resolve collisions for player and box
        player.StopByWalls(walls);
        box.StopByWalls(walls);

        // Update sections
        player.Update(gameTime);
        box.Update(gameTime);
        ground.Update(gameTime);

        base.Update(gameTime);
    }

    protected override void Draw(GameTime gameTime)
    {
        GraphicsDevice.Clear(Color.CornflowerBlue);

        // Draw the player
        _spriteBatch.Begin();
        player.Draw(_spriteBatch);
        box.Draw(_spriteBatch);
        ground.Draw(_spriteBatch);
        platform.Draw(_spriteBatch);
        _spriteBatch.End();

        base.Draw(gameTime);
    }
}
